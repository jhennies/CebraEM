
from glob import glob
import os
import numpy as np
from tifffile import imread, imsave
from pybdv import make_bdv
from pybdv.util import open_file, get_key
from h5py import File

from .data import (
    small_objects_to_zero,
    relabel_consecutive,
    crop_zero_padding_3d,
    scale_and_shift,
    quantile_norm
)
from .bdv_utils import create_empty_dataset


SUPPORTED_FILE_TYPES = ['h5', 'n5', 'model']


def _clip_values(im, values):
    im[im > values[1]] = values[1]
    im = (im - values[0]).astype('float16')
    im = (im / ((values[1] - values[0]) / 255)).astype('uint8')
    return im


def read_volume_from_tif_stack(in_path, roi, clip_values, verbose=False):

    if roi is not None:
        xyr = np.s_[roi[1]: roi[1] + roi[4], roi[0]: roi[0] + roi[3]]
        zr = np.s_[roi[2]: roi[2] + roi[5]]
    else:
        xyr = np.s_[:]
        zr = np.s_[:]

    im_list = sorted(glob(os.path.join(in_path, '*.tif')))[zr]
    assert im_list, 'Folder does not contain any tif files!'
    ims = []
    for filepath in im_list:
        if verbose:
            print('Reading {}'.format(filepath))
        im = imread(filepath)[xyr][None, :]
        if clip_values is not None:
            im = _clip_values(im, clip_values)
        ims.append(im)
    return np.concatenate(ims, axis=0)


def read_volume_from_container(in_filepath, roi, clip_values, key, axes_order='zyx'):

    if axes_order == 'zyx':
        axes = dict(z=0, y=1, x=2)
    elif axes_order == 'xyz':
        axes = dict(x=0, y=1, z=2)
    else:
        raise NotImplementedError(f'Axes order {axes_order} not implemented!')

    axes_order_list = list(axes_order)

    if roi is not None:
        roi = tuple([np.s_[roi[axes[a]]: roi[axes[a]] + roi[axes[a] + 3]] for a in axes_order_list])
    else:
        roi = np.s_[:]

    print(f'roi = {roi}')

    # FIXME this doesn't work
    try:
        with open_file(in_filepath, mode='r') as f:
            data = f[key][roi]
    except ValueError:
        with File(in_filepath, mode='r') as f:
            data = f[key][roi]

    if clip_values is not None:
        data = _clip_values(data, clip_values)

    if axes_order != 'zyx':
        data = np.transpose(data, axes=[
            axes_order_list.index('z'),
            axes_order_list.index('y'),
            axes_order_list.index('x')
        ])

    return data


def apply_size_filter(
        volume,
        filter_size
):

    # Get unique values
    u, c = np.unique(volume, return_counts=True)

    # Find which labels to remove
    to_remove_ids = np.where(c < filter_size)[0]

    # Put these labels to zero
    for idx in to_remove_ids:
        volume[volume == u[idx]] = 0

    return volume


def connected_components_analysis(volume):
    from vigra.analysis import labelVolumeWithBackground, relabelConsecutive
    volume = labelVolumeWithBackground(volume.astype('uint32'), background_value=0)
    volume = relabel_consecutive(volume, sort_by_size=True)
    # volume = relabelConsecutive(volume, keep_zeros=True)[0]
    return volume


def optimize_data_type(volume):
    max_val = np.max(volume)
    if max_val < 2 ** 8:
        volume = volume.astype('uint8')
    elif max_val < 2 ** 16:
        volume = volume.astype('uint16')
    elif max_val < 2 ** 32:
        volume = volume.astype('uint32')
    else:
        raise OverflowError('Maximum value exceeds implemented data types')
    return volume


def convert_to_bdv(
        source_path,
        target_path,
        key=None,
        resolution=(0.01, 0.01, 0.01),
        unit='micrometer',
        clip_values=None,
        invert=False,
        roi=None,
        bdv_scale_factors=(2, 2, 4),
        scale_mode='mean',
        connected_components=False,
        size_filter=0,
        axes_order='zyx',
        verbose=False
):

    # Currently CebraEM internally only uses micrometer, hence I am only supporting it here as well
    if unit != 'micrometer':
        print('\nCebraEM uses the pixel spacing in micrometers. Hence only micrometer is supported here.\n'
              'Re-run the function and supply the resolution in micrometer scale!\n\n'
              'Exiting ...\n')
        return

    # Checking for the file type
    file_type = os.path.splitext(source_path)[1]

    if file_type == '.model' and scale_mode != 'nearest':
        scale_mode = 'nearest'
        print('Info: Set scale mode to nearest!')

    # Load data (use a ROI if specified)
    print('Reading data ...')
    if os.path.isdir(source_path) and os.path.splitext(source_path)[1] != '.n5':
        # Assuming a directory with tif files
        volume = read_volume_from_tif_stack(source_path, roi, clip_values, verbose=verbose)
    else:
        if verbose:
            print(f'Found {file_type} file extension')
        if file_type == '.n5' or file_type == '.h5':
            assert key is not None
            volume = read_volume_from_container(source_path, roi, clip_values, key, axes_order=axes_order)
        elif file_type == '.model':
            if key is None:
                key = 'mibModel'
            volume = read_volume_from_container(source_path, roi, clip_values, key, axes_order='zxy')

    if invert:
        print('Inverting data ...')
        assert volume.dtype == 'uint8', 'Currently only supporting uint8 data type'
        volume = 255 - volume

    if verbose:
        print(f'volume.shape = {volume.shape}')
        print(f'volume.dtype = {volume.dtype}')

    if connected_components:
        print(f'Running connected components analysis ...')
        volume = connected_components_analysis(volume)
    if size_filter > 0:
        print(f'Applying size filter ...')
        # volume = apply_size_filter(volume, size_filter)
        volume = small_objects_to_zero(volume, size_filter, verbose=verbose)

    if connected_components or size_filter:
        volume = optimize_data_type(volume)

    if verbose:
        print('volume.shape = {}'.format(volume.shape))

    scale_factors = [[sc] * 3 for sc in bdv_scale_factors]
    if verbose:
        print('scale_factors = {}'.format(scale_factors))

    print('Generating BDV format ...')
    make_bdv(volume, target_path,
             downscale_factors=scale_factors,
             downscale_mode=scale_mode,
             resolution=resolution, unit=unit)


def normalize_instances(
        raw_source,
        raw_key,
        seg_source,
        seg_key,
        target_path,
        raw_resolution=(0.01, 0.01, 0.01),
        seg_resolution=(0.01, 0.01, 0.01),
        unit='micrometer',
        instance_ids=None,
        quantiles=(0.1, 0.9),
        anchor_values=(0.05, 0.95),
        dtype='uint8',
        bdv_scale_factors=(2, 2, 4),
        scale_mode='mean',
        verbose=False,
):

    print(f'Fetching inputs ...')
    # Read the segmentation
    with open_file(seg_source, mode='r') as f:
        seg = f[seg_key][:]
    # Open the raw dataset (but don't read the data at once!)
    rf = open_file(raw_source, mode='r')[raw_key]

    if verbose:
        print(f'seg.shape = {seg.shape}')
        print(f'rf.shape = {rf.shape}')

    bdv_scale_factors = [[x] * 3 for x in bdv_scale_factors]

    print('Generating target dataset ...')
    # Generate a target dataset of the same shape as the raw dataset
    create_empty_dataset(
        target_path,
        0, 0,
        rf.shape,
        data_dtype=dtype,
        scale_factors=bdv_scale_factors,
        resolution=raw_resolution,
        unit=unit,
        verbose=verbose
    )

    # Get the target dataset handle
    target_key = get_key(False, 0, 0, 0)
    tfh = open_file(target_path, mode='a')

    # Get the IDs of the instances
    if instance_ids is None:
        print('Fetching instance ids ...')
        instance_ids = np.unique(seg)[1:]

    if verbose:
        print(f'instance_ids = {instance_ids}')

    seg_resolution = np.array(seg_resolution)
    raw_resolution = np.array(raw_resolution)

    for idx in instance_ids:

        # --- Get all pixels of the current object in the raw data ---

        # Get the bounding box of the current object in the segmentation
        bounds, top_left, bottom_right = crop_zero_padding_3d(seg == idx, return_as_arrays=True)

        # Fetch the data from the segmentation
        this_obj = seg[bounds]
        this_obj[this_obj != idx] = 0

        # Transorm the bounds to the raw resolution
        scale = seg_resolution / raw_resolution
        top_left_rr = (top_left * scale).astype(int)
        bottom_right_rr = ((bottom_right + 1) * scale).astype(int)

        if verbose:
            print(f'top_left = {top_left}')
            print(f'top_left_rr = {top_left_rr}')
            print(f'bottom_right = {bottom_right}')
            print(f'bottom_right_rr = {bottom_right_rr}')

        # Fetch the object area from the raw data
        bounds_rr = np.s_[top_left_rr[0]:bottom_right_rr[0],
                 top_left_rr[1]:bottom_right_rr[1],
                 top_left_rr[2]:bottom_right_rr[2]]
        this_raw = rf[bounds_rr]

        # Scale the seg to match the raw
        if verbose:
            print(f'scale = {scale}')
        this_obj_r = scale_and_shift(
            this_obj,
            scale=scale,
            scale_im_size=this_raw.shape,
            order=0
        )

        if verbose:
            print(f'this_obj_r.shape = {this_obj_r.shape}')
            print(f'this_raw.shape = {this_raw.shape}')
        assert this_obj_r.shape == this_raw.shape

        # from matplotlib import pyplot as plt
        # plt.imshow(this_obj_r[0, :])
        # plt.figure()
        # plt.imshow(this_raw[0, :])
        # plt.show()

        this_raw[this_obj_r > 1] = quantile_norm(this_raw[this_obj_r > 1], quantiles[0], quantiles[1], verbose=verbose)

        if verbose:
            print(f'np.unique(this_raw) = {np.unique(this_raw)}')
        tfh[target_key][bounds_rr] = this_raw
    tfh.close()
