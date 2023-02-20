
import numpy as np
import scipy.ndimage as ndi
from pybdv.util import open_file
from .alignment import xcorr_on_volume


def _apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.,
                    ndim=3,
                    order=1):
    """Apply the image transformation specified by a matrix.

    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:ndim, :ndim]
    final_offset = transform_matrix[:ndim, ndim]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=order,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def _transform_matrix_offset_center(matrix, x, y, z):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    o_z = float(z) / 2 + 0.5
    offset_matrix = np.array([[1, 0, 0, o_x],
                              [0, 1, 0, o_y],
                              [0, 0, 1, o_z],
                              [0, 0, 0, 1]])
    reset_matrix = np.array([[1, 0, 0, -o_x],
                             [0, 1, 0, -o_y],
                             [0, 0, 1, -o_z],
                             [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def _scale_and_shift(vol, scale, shift, order=1, scale_im_size=False, verbose=False):

    im = vol[..., None]

    if scale_im_size:
        new_shape = np.array(im.shape)
        if scale[0] > 1:
            new_shape[0] = int(im.shape[0] * scale[0])
        if scale[1] > 1:
            new_shape[1] = int(im.shape[1] * scale[1])
        if scale[2] > 1:
            new_shape[2] = int(im.shape[2] * scale[2])
        im_new = np.zeros(new_shape, dtype=im.dtype)
        if verbose:
            print(f'im_new.shape = {im_new.shape}')
        pos_in_new = ((np.array(new_shape) - np.array(im.shape)) / 2).astype(int)
        im_new[
            pos_in_new[0]:pos_in_new[0] + im.shape[0],
            pos_in_new[1]:pos_in_new[1] + im.shape[1],
            pos_in_new[2]:pos_in_new[2] + im.shape[2],
            :
        ] = im
        im = im_new

    if verbose:
        print(f'im.shape = {im.shape}')

    transform_matrix = np.array([[1/scale[0], 0, 0, 0],
                                 [0, 1/scale[1], 0, 0],
                                 [0, 0, 1/scale[2], 0],
                                 [0, 0, 0, 1]])

    shift_matrix = np.array([[1, 0, 0, shift[0]],
                             [0, 1, 0, shift[1]],
                             [0, 0, 1, shift[2]],
                             [0, 0, 0, 1]])

    transform_matrix = np.dot(shift_matrix, transform_matrix)

    if verbose:
        print(f'transform_matrix = {transform_matrix}')

    fill_mode = 'nearest'
    cval = 0.
    w, h, d = im.shape[2], im.shape[1], im.shape[0]
    transform_matrix = _transform_matrix_offset_center(transform_matrix, d, h, w)
    im = _apply_transform(im, transform_matrix, 3,
                          fill_mode=fill_mode, cval=cval, order=order)

    if scale_im_size:
        new_shape = np.array(im.shape)
        if scale[0] < 1:
            new_shape[0] = int(im.shape[0] * scale[0])
        if scale[1] < 1:
            new_shape[1] = int(im.shape[1] * scale[1])
        if scale[2] < 1:
            new_shape[2] = int(im.shape[2] * scale[2])
        new_pos = ((np.array(im.shape) - np.array(new_shape)) / 2).astype(int)
        im = im[
             new_pos[0]: new_pos[0] + new_shape[0],
             new_pos[1]: new_pos[1] + new_shape[1],
             new_pos[2]: new_pos[2] + new_shape[2],
        ]
    return im.squeeze()


def load_with_zero_padding(dataset, starts, ends, shape, verbose=False):

    starts = np.array(starts)
    ends = np.array(ends)
    shape = np.array(shape)

    starts_target = np.zeros((3,), dtype=int)
    starts_target[starts < 0] = -starts[starts < 0]
    starts_source = np.zeros((3,), dtype=int)
    starts_source[starts > 0] = starts[starts > 0]

    too_large = (np.array(dataset.shape) - ends) < 0
    ends_target = shape.copy()
    ends_target[too_large] = (np.array(dataset.shape) - starts)[too_large]
    ends_source = ends.copy()
    ends_source[too_large] = np.array(dataset.shape)[too_large]

    if verbose:
        print(f'shape = {shape}')
        print(f'dataset.shape = {dataset.shape}')
        print(f'starts = {starts}')
        print(f'ends = {ends}')
        print(f'starts_source = {starts_source}')
        print(f'starts_target = {starts_target}')
        print(f'ends_source = {ends_source}')
        print(f'ends_target = {ends_target}')

    slicing_source = np.s_[
        starts_source[0]: ends_source[0],
        starts_source[1]: ends_source[1],
        starts_source[2]: ends_source[2]
    ]
    slicing_target = np.s_[
        starts_target[0]: ends_target[0],
        starts_target[1]: ends_target[1],
        starts_target[2]: ends_target[2]
    ]

    vol = np.zeros(shape, dtype=dataset.dtype)
    vol[slicing_target] = dataset[slicing_source]

    return vol


def load_data(
        input_path,
        internal_path,
        pos, shape,
        xcorr=False,
        verbose=False
):
    shape = np.array(shape)
    with open_file(input_path, mode='r') as f:

        if verbose:
            print('dataset_shape = {}'.format(f[internal_path].shape))
            print(f'pos = {pos}')
            print(f'shape = {shape}')
        vol = load_with_zero_padding(f[internal_path], pos, pos + shape, shape, verbose=verbose)
        if verbose:
            print('extracted_shape = {}'.format(vol.shape))

        if xcorr:
            print('Re-alignment on extracted volume using xcorr ...')
            vol = xcorr_on_volume(vol)

    return vol


def crop_and_scale(
        input_path,
        position,
        internal_path,
        input_res=(10, 10, 10),
        output_res=None,
        natural_out_res=False,
        output_shape=(512, 512, 512),
        scale_result=True,
        order=1,
        xcorr=False,
        verbose=False,
):
    """
    Extracts and scales a volume from a dataset to match a certain output shape and resolution

    :param input_path: The source dataset
    :param position:
    :param internal_path: Path in the source dataset
    :param input_res:
    :param output_res:
    :param natural_out_res: Natural isotropic counterpart of input_res is used for output_res; overwrites output_res
    :param output_shape:
    :param scale_result:
    :param order:
    :param xcorr:
    :param verbose:
    :return:
    """

    if natural_out_res and output_res is not None:
        print('Warning: overwriting output res')
    if natural_out_res:
        output_res = [np.prod(input_res) ** (1 / 3)] * 3
    if output_res is None:
        output_res = input_res
    output_res = np.array(output_res)
    input_res = np.array(input_res)
    position = np.array(position)
    output_shape = np.array(output_shape)

    p0 = position
    p1 = position + output_shape
    p0_dash = p0 * output_res / input_res
    p1_dash = p1 * output_res / input_res

    floor_p0_dash = np.floor(p0_dash).astype(int)
    ceil_p1_dash = np.ceil(p1_dash).astype(int)

    input_shape = np.max([p1 - p0, ceil_p1_dash - floor_p0_dash], axis=0)

    scale = input_res / output_res

    shift = 0.5 / scale * (input_shape - scale * input_shape) \
        + (p0_dash - floor_p0_dash)

    # Load raw data
    raw = load_data(input_path, internal_path, floor_p0_dash, input_shape, xcorr=xcorr, verbose=verbose)
    if scale_result:
        if np.sum(scale != 1) or np.sum(shift != 0):
            raw = _scale_and_shift(raw, scale, shift, order=order, verbose=verbose)
            raw = raw[:output_shape[0], :output_shape[1], :output_shape[2]]

    return raw


def quantile_norm(volume, qlow, qhigh):

    dtype = volume.dtype
    assert dtype == 'uint8', 'Only unsigned 8bit is implemented'
    volume = volume.astype('float64')

    # Get quantiles of full volume
    # Could potentially also be a reference slice, multiple reference slices, ...
    q_lower_ref = np.quantile(volume, qlow)
    q_upper_ref = np.quantile(volume, qhigh)

    volume -= q_lower_ref
    volume /= q_upper_ref - q_lower_ref
    volume *= 255

    # Clip everything that went out of range
    # FIXME this assumes dtype==uint8
    volume[volume < 0] = 0
    volume[volume > 255] = 255

    # Convert back to the original dtype
    return volume.astype(dtype)
