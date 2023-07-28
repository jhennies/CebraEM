
import sys
import numpy as np
import os
import scipy.ndimage as ndi
from pybdv.util import open_file
from cebra_em_core.dataset.alignment import xcorr_on_volume
from concurrent.futures import ThreadPoolExecutor


def crop_zero_padding_3d(dat, return_as_arrays=False, add_halo=None):

    max_shape = np.array(dat.shape)
    # argwhere will give you the coordinates of every non-zero point
    true_points = np.argwhere(dat)

    if add_halo is None:
        # take the smallest points and use them as the top left of your crop
        top_left = true_points.min(axis=0)
        # take the largest points and use them as the bottom right of your crop
        bottom_right = true_points.max(axis=0)
    else:
        # take the smallest points and use them as the top left of your crop
        top_left = true_points.min(axis=0) - add_halo
        top_left[top_left < 0] = 0
        # take the largest points and use them as the bottom right of your crop
        bottom_right = true_points.max(axis=0) + add_halo
        bottom_right[bottom_right > max_shape] = max_shape[bottom_right > max_shape]

    # generate bounds
    bounds = np.s_[top_left[0]:bottom_right[0] + 1,  # plus 1 because slice isn't
                   top_left[1]:bottom_right[1] + 1,  # inclusive
                   top_left[2]:bottom_right[2] + 1]
    if return_as_arrays:
        return bounds, top_left, bottom_right
    return bounds


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
    o_x = float(x) / 2  # + 0.5
    o_y = float(y) / 2  # + 0.5
    o_z = float(z) / 2  # + 0.5
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


def scale_and_shift(vol, scale, shift=(0., 0., 0.), order=1, scale_im_size=False, verbose=False):

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
        pos,
        shape,
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
        extended_return=False,
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
    :param extended_return: Return transformations as well
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
    # TODO look at the division again
    p0_dash = p0 * output_res / input_res.astype(float)
    p1_dash = p1 * output_res / input_res.astype(float)

    # Round p0_dash and p1_dash to two decimals
    #  sometimes they are ever so slightly below the actual value: e.g. 0.999999999... instead of 1.0
    #  which leads the floor operation below to land on the wrong value
    if verbose:
        print(f'p0_dash = {p0_dash}')
        print(f'p1_dash = {p1_dash}')
    p0_dash = np.round(p0_dash, decimals=2)
    p1_dash = np.round(p1_dash, decimals=2)
    # assert (p0_dash * 100 == (p0_dash * 100).astype(int)).all(), f'p0_dash must not be float: p0_dash = {p0_dash}'
    # assert (p1_dash * 100 == (p1_dash * 100).astype(int)).all(), f'p1_dash must not be float: p1_dash = {p1_dash}'

    floor_p0_dash = np.floor(p0_dash).astype(int)
    rem_p0_dash = p0_dash - floor_p0_dash
    ceil_p1_dash = np.ceil(p1_dash).astype(int)

    input_shape = ceil_p1_dash - floor_p0_dash
    scale = input_res / output_res

    # This copes with scaling that leads to subpixel positions
    shift = rem_p0_dash

    if verbose:
        print(f'input_res = {input_res}')
        print(f'output_res = {output_res}')
        print(f'p0 = {p0}')
        print(f'p1 = {p1}')
        print(f'p0_dash = {p0_dash}')
        print(f'p1_dash = {p1_dash}')
        print(f'floor_p0_dash = {floor_p0_dash}')
        print(f'rem_p0_dash = {rem_p0_dash}')
        print(f'ceil_p1_dash = {ceil_p1_dash}')
        print(f'input_shape = {input_shape}')
        print(f'scale = {scale}')
        print(f'shift = {shift}')

    # Load raw data
    raw = load_data(
        input_path,
        internal_path,
        floor_p0_dash,
        input_shape,
        xcorr=xcorr,
        verbose=verbose
    )
    if scale_result:
        if np.sum(scale != 1) or np.sum(shift != 0):
            raw = scale_and_shift(raw, scale, shift, scale_im_size=True, order=order, verbose=verbose)
            raw = raw[:output_shape[0], :output_shape[1], :output_shape[2]]

    assert np.sum(np.array(raw.shape) - np.array(output_shape)) == 0, (
        f'Shape of the computed map and expected output '
        f'shape must match! raw.shape = {raw.shape} != output_shape = {output_shape}'
    )

    if extended_return:
        return raw, scale, shift
    return raw


def quantile_norm(volume, qlow, qhigh, verbose=False):

    dtype = volume.dtype
    assert dtype == 'uint8', 'Only unsigned 8bit is implemented'
    volume = volume.astype('float64')

    # Get quantiles of full volume
    # Could potentially also be a reference slice, multiple reference slices, ...
    if verbose:
        print(f'qlow = {qlow}')
        print(f'np.unique(volume) = {np.unique(volume)}')
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


def small_objects_to_zero(m, size_filter, verbose=False, n_workers=os.cpu_count()):

    def _to_zero(idx, obj_id):
        sys.stdout.write('\r' + 'Identifying small objects: {} %'.format(int(100 * float(idx + 1) / float(len(smalls)))))
        m[m == obj_id] = 0

    if verbose:
        print('Finding small objects ...')
    u, c = np.unique(m, return_counts=True)
    smalls = u[c < size_filter]

    if verbose:
        print('Setting smalls to zero ...')

    with ThreadPoolExecutor(max_workers=n_workers) as tpe:
        tasks = [
            tpe.submit(_to_zero, idx, small)
            for idx, small in enumerate(smalls)
        ]
        [task.result() for task in tasks]

    return m


def relabel_consecutive(map, sort_by_size=False, n_workers=os.cpu_count()):

    def _relabel(idx, label, segment):
        sys.stdout.write('\r' + 'Relabelling: {} %'.format(int(100 * float(idx + 1) / float(len(relabel_dict)))))
        # print('label {} -> segment {}'.format(label, segment))
        map[map == label] = -segment

    map = map + 1

    if sort_by_size:
        labels, counts = np.unique(map, return_counts=True)
        labels = labels[np.argsort(counts)[::-1]].tolist()
    else:
        labels = np.unique(map).tolist()
    relabel_dict = dict(zip(labels, range(len(labels))))

    # Perform the mapping
    c = 0
    map = map.astype('float32')
    with ThreadPoolExecutor(max_workers=n_workers) as tpe:
        tasks = []
        idx = 0
        for label, segment in relabel_dict.items():
            tasks.append(
                tpe.submit(_relabel, idx, label, segment)
            )
            idx += 1
        [task.result() for task in tasks]
    map = (-map).astype('float32')

    return map


def get_quantiles(
        raw_handle,
        seg,
        raw_resolution,
        seg_resolution,
        seg_ids=None,
        quantile_spacing=0.2,
        method='dense',
        pixels_per_object=1024,
        debug=False,
        verbose=False
):

    if seg is not None:
        if seg_ids is None:
            seg_ids = np.unqiue(seg)[1:]
    else:
        seg_ids = [0]

    raw_resolution, seg_resolution = np.array(raw_resolution), np.array(seg_resolution)
    quantiles = {}

    for idx in seg_ids:

        # --- Get all pixels of the current object in the raw data ---
        print(f'Extracting quantiles for idx = {idx}')

        # Get the bounding box of the current object in the segmentation
        if seg is not None:
            bounds, top_left, bottom_right = crop_zero_padding_3d(seg == idx, return_as_arrays=True)

            # Fetch the data from the segmentation
            this_obj = seg[bounds]
            this_obj[this_obj != idx] = 0

            scale = seg_resolution / raw_resolution

            if debug:
                if not os.path.exists('tmp'):
                    os.mkdir('tmp')
                print(os.getcwd())
                with open_file(f'tmp/mask_{idx}.h5', mode='w') as f:
                    f.create_dataset('data', data=this_obj, compression='gzip')

            if verbose:
                print(f'top_left = {top_left} | bottom_right = {bottom_right}')
                print(f'seg_resolution = {seg_resolution}')
                print(f'raw_resolution = {raw_resolution}')
                print(f'scale = {scale}')
        else:
            scale = 1.
            top_left = np.array((0, 0, 0))

        if method == 'dense':

            assert seg is not None, 'Not implemented for no mask!'

            # Transorm the bounds to the raw resolution
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
            this_raw = raw_handle[bounds_rr]

            # Get the raw pixels of the current object
            print(f'Fetching the raw pixels ...')
            pos = np.where(this_obj >= 1)
            pos = (
                (pos[0] * scale[0]).astype(int),
                (pos[1] * scale[1]).astype(int),
                (pos[2] * scale[2]).astype(int)
            )
            raw_pixels = this_raw[pos]

        elif method == 'sparse':

            # Sample the points from within the mask
            if seg is not None:
                pos = np.argwhere(this_obj >= 1)
            else:
                # Make a grid of points on the raw data
                pos = np.argwhere(np.ones((np.array(raw_handle.shape) / 8).astype(int)) == 1) * 8
            rand_ids = np.random.randint(len(pos), size=pixels_per_object)
            pos = pos[rand_ids, :]

            def extract_pixel_values(p):
                try:
                    return raw_handle[tuple((np.array(p) + top_left) * scale)]
                except ValueError:
                    # This happens when the position is out of bounds in the raw data (due to scaling issues)
                    return None

            # Extract the pixel values

            # raw_pixels = np.array(
            #     [
            #         raw_handle[tuple((np.array(p) + top_left) * scale)]
            #         for p in pos
            #     ]
            # )
            raw_pixels = [extract_pixel_values(p) for p in pos]
            raw_pixels = np.array([px for px in raw_pixels if px is not None])
            if len(raw_pixels) < pixels_per_object:
                print(f'Warning: The number of extracted pixels is smaller than requested: '
                      f'{len(raw_pixels)} < {pixels_per_object}')

            if verbose:
                print(f'raw_pixels = {raw_pixels}')

        else:
            raise ValueError(f'Invalid normalization method: {method}')

        # Determine the quantile list
        if verbose:
            print('Computing the quantiles ...')
        this_quantiles = {q: np.quantile(raw_pixels, q / 100) for q in range(0, 100, int(quantile_spacing * 100))}

        # Append to the return dict
        quantiles[idx] = this_quantiles

    return quantiles
