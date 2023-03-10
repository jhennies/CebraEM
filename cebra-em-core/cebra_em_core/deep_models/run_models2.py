
import numpy as np
import h5py
import os
import sys
from vigra.filters import gaussianSmoothing

from cebra_em_core.dataset.data import quantile_norm

try:
    import torch as t
except ModuleNotFoundError:
    print('torch not available!')


def pre_processing(raw, sigma=0.0, qnorm_low=0.0, qnorm_high=1.0):

    if qnorm_low > 0.0 or qnorm_high < 1.0:
        raw = quantile_norm(raw, qnorm_low, qnorm_high)

    if sigma > 0:
        raw = gaussianSmoothing(raw, sigma)

    return raw


def write_test_h5_generator_result(dataset, result, x, y, z, overlap, ndim=3):

    size = result.shape[1: ndim+1]
    if ndim == 2:
        size = (1,) + size
    dataset_shape = dataset.shape

    p = (z, y, x)

    s_ds = []
    s_r = []

    for idx in range(0, 3):
        zds = []
        zr = []
        if p[idx] == 0:
            zds.append(0)
            zr.append(0)
        else:
            zds.append(int(p[idx] + overlap[idx] / 2))
            zr.append(int(overlap[idx] / 2))

        if p[idx] + size[idx] == dataset_shape[idx]:
            zds.append(None)
            zr.append(None)
        else:
            zds.append(int(p[idx] + size[idx] - overlap[idx] / 2))
            zr.append(-int(overlap[idx] / 2))
            if zr[-1] == 0:
                zr[-1] = None

        s_ds.append(zds)
        s_r.append(zr)

    result = result.squeeze()
    if ndim == 2:
        result = result[None, :]

    if result.ndim == 3:
        # Just one channel

        dataset[s_ds[0][0]:s_ds[0][1], s_ds[1][0]:s_ds[1][1], s_ds[2][0]:s_ds[2][1]] \
            = (result * 255).astype('uint8')[s_r[0][0]:s_r[0][1], s_r[1][0]:s_r[1][1], s_r[2][0]:s_r[2][1]]

    elif result.ndim == 4:
        # Multiple channels

        dataset[s_ds[0][0]:s_ds[0][1], s_ds[1][0]:s_ds[1][1], s_ds[2][0]:s_ds[2][1], :] \
            = (result * 255).astype('uint8')[s_r[0][0]:s_r[0][1], s_r[1][0]:s_r[1][1], s_r[2][0]:s_r[2][1], :]


def _build_equally_spaced_volume_list(
        spacing,
        area_size,
        n_volumes,
        transform_ratio,
        set_volume=None,
        verbose=False
):

    # Components
    spacing = spacing
    half_area_size = (np.array(area_size) / 2).astype(int)

    # This generates the list of all positions, equally spaced and centered around zero
    mg = np.mgrid[
         -half_area_size[0]: half_area_size[0]: spacing[0],
         -half_area_size[1]: half_area_size[1]: spacing[1],
         -half_area_size[2]: half_area_size[2]: spacing[2]
    ].squeeze()
    mg[0] -= int((mg[0].max() + mg[0].min()) / 2)
    mg[1] -= int((mg[1].max() + mg[1].min()) / 2)
    mg[2] -= int((mg[2].max() + mg[2].min()) / 2)
    mg = mg.reshape(3, np.prod(np.array(mg.shape)[1:]))
    positions = mg.swapaxes(0, 1)

    n_transform = int(n_volumes * len(positions) * transform_ratio)
    transform = [True] * n_transform + [False] * (n_volumes * len(positions) - n_transform)
    np.random.shuffle(transform)

    index_array = []

    idx = 0
    for volume in range(n_volumes):
        for position in positions:

            if set_volume:
                index_array.append(
                    [
                        position,
                        set_volume,          # Always volume 0
                        transform[idx]
                    ]
                )
            else:
                index_array.append(
                    [
                        position,
                        volume,          # Always volume 0
                        transform[idx]
                    ]
                )
            idx += 1

    if verbose:
        print('Equally spaced volumes:')
        print('    Total samples:       {}'.format(len(positions) * n_volumes))
        print('    Volumes:             {}'.format(n_volumes))
        print('    Transformed samples: {}'.format(n_transform))
        print('Actual size of index_array: {}'.format(len(index_array)))

    return index_array


def _find_bounds(position, crop_shape, full_shape):
    position = np.array(position)
    crop_shape = np.array(crop_shape)
    full_shape = np.array(full_shape)

    # Start and stop in full volume (not considering volume boundaries)
    start = (position - crop_shape / 2 + full_shape / 2).astype('int16')
    stop = start + crop_shape

    # Checking for out of bounds
    start_corrected = start.copy()
    start_corrected[start < 0] = 0
    start_oob = start_corrected - start
    stop_corrected = stop.copy()
    stop_corrected[stop > full_shape] = full_shape[stop > full_shape]
    stop_oob = stop - stop_corrected

    # Making slicings ...
    # ... where to take the data from in the full shape ...
    s_source = np.s_[
               start_corrected[0]: stop_corrected[0],
               start_corrected[1]: stop_corrected[1],
               start_corrected[2]: stop_corrected[2]
               ]
    # ... and where to put it into the crop
    s_target = np.s_[
               start_oob[0]: crop_shape[0] - stop_oob[0],
               start_oob[1]: crop_shape[1] - stop_oob[1],
               start_oob[2]: crop_shape[2] - stop_oob[2]
               ]

    return s_source, s_target


def _load_data_with_padding(
        channels,
        position,
        target_shape,
        auto_pad=False,
        return_pad_mask=False,
        return_shape_only=False,
        auto_pad_z=False
):
    source_shape = np.array(channels[0].shape)

    shape = np.array(target_shape)
    if auto_pad:
        shape[1:] = np.ceil(np.array(target_shape[1:]) * np.sqrt(2) / 2).astype(int) * 2
    if auto_pad_z:
        shape[0] = np.ceil(np.array(target_shape[0]) * np.sqrt(2) / 2).astype(int) * 2

    if return_shape_only:
        return shape.tolist() + [len(channels)]

    s_source, s_target = _find_bounds(position, shape, source_shape)

    # Defines the position of actual target data within the padded data
    pos_in_pad = ((shape - target_shape) / 2).astype(int)
    s_pos_in_pad = np.s_[pos_in_pad[0]: pos_in_pad[0] + target_shape[0],
                   pos_in_pad[1]: pos_in_pad[1] + target_shape[1],
                   pos_in_pad[2]: pos_in_pad[2] + target_shape[2]]

    x = []
    for cid, channel in enumerate(channels):
        # Load the data according to the definitions above

        vol_pad = np.zeros(shape, dtype=channel.dtype)

        vol_pad[s_target] = channel[s_source]
        x.append(vol_pad[..., None])

    if return_pad_mask:
        pad_mask = np.zeros(x[0].shape, dtype=channels[0].dtype)
        pad_mask[s_target] = 255
        x.append(pad_mask)

    x = np.concatenate(x, axis=3)

    return x, s_pos_in_pad


def _write_result(dataset, result, position, spacing):

    spacing = np.array(spacing)
    spacing_half = (spacing / 2).astype(int)
    shape = np.array(dataset.shape[:3])
    shape_half = (shape / 2).astype(int)
    result_shape = np.array(result.shape[:3])
    result_shape_half = (result_shape / 2).astype(int)

    # Pre-crop the result
    start_crop = result_shape_half - spacing_half
    stop_crop = result_shape_half + spacing_half
    s_pre_crop = np.s_[
        start_crop[0]: stop_crop[0],
        start_crop[1]: stop_crop[1],
        start_crop[2]: stop_crop[2]
    ]
    result_cropped = result[s_pre_crop]

    # All the shapes and positions
    result_shape = np.array(result_cropped.shape[:3])
    result_shape_half = (result_shape / 2).astype(int)
    position = np.array(position)

    start_pos = position + shape_half - result_shape_half
    stop_pos = start_pos + spacing
    # print('')
    # print('Before correction ...')
    # print('start_pos = {}'.format(start_pos))
    # print('stop_pos = {}'.format(stop_pos))
    start_out_of_bounds = np.zeros(start_pos.shape, dtype=start_pos.dtype)
    start_out_of_bounds[start_pos < 0] = start_pos[start_pos < 0]
    stop_out_of_bounds = stop_pos - shape
    stop_out_of_bounds[stop_out_of_bounds < 0] = 0
    start_pos[start_pos < 0] = 0
    stop_pos[stop_out_of_bounds > 0] = shape[stop_out_of_bounds > 0]
    # print('After correction ...')
    # print('start_pos = {}'.format(start_pos))
    # print('stop_pos = {}'.format(stop_pos))

    # For the results volume
    s_source = np.s_[
               -start_out_of_bounds[0]:stop_pos[0] - start_pos[0] - start_out_of_bounds[0],
               -start_out_of_bounds[1]:stop_pos[1] - start_pos[1] - start_out_of_bounds[1],
               -start_out_of_bounds[2]:stop_pos[2] - start_pos[2] - start_out_of_bounds[2],
               ]
    # For the target dataset
    s_target = np.s_[
               start_pos[0]:stop_pos[0],
               start_pos[1]:stop_pos[1],
               start_pos[2]:stop_pos[2]
               ]

    dataset[s_target] = (result_cropped * 255).astype('uint8')[s_source]


def transform_matrix_offset_center(matrix, x, y, z):
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


def predict_model_from_h5(
        model,  # The model has to be fully ready for computation (basically a function that yields the result)
        results_filepath,
        raw_channels,
        num_result_channels,
        target_size,
        overlap,
        scale=1.,
        compute_empty_volumes=True,
        thresh=(0, 255),
        use_compute_map=False,
        downsample_compute_map=None,
        mask=None,
        squeeze_result=False,
        verbose=False
):

    size = list(raw_channels[0][0].shape)
    result_size = (np.array(size) * scale).tolist()
    if not squeeze_result:
        result_size = result_size + [num_result_channels]
    else:
        if num_result_channels > 1:
            result_size = result_size + [num_result_channels]
    size = size + [num_result_channels]

    if mask is not None:
        use_compute_map = True
        if downsample_compute_map:
            assert list(mask.shape) == (np.array(size[:3]) / downsample_compute_map).astype(int).tolist()
        else:
            assert list(mask.shape) == size[:3]

    if verbose:
        print('Generating results file ...')
    if use_compute_map:
        if not os.path.exists(results_filepath):
            # Generate results file
            with h5py.File(results_filepath, 'w') as f:
                f.create_dataset('data', shape=result_size, dtype='uint8', compression='gzip')
                if downsample_compute_map:
                    f.create_dataset('compute_map', shape=(np.array(size[:3]) / downsample_compute_map).astype(int), dtype='bool', compression='gzip')
                else:
                    f.create_dataset('compute_map', shape=size[:3], dtype='bool', compression='gzip')
                # if scale != 1:
                #     f.create_dataset('scaled_input', shape=result_size, dtype='uint8', compression='gzip')
    else:
        # Generate results file
        with h5py.File(results_filepath, 'w') as f:
            f.create_dataset('data', shape=result_size, dtype='uint8', compression='gzip')
            # if scale != 1:
            #     f.create_dataset('scaled_input', shape=result_size, dtype='uint8', compression='gzip')

    if verbose:
        print('Writing mask to compute map ...')
    if mask is not None:
        # FIXME This requires that a maximum filter (box) is performed on the mask first
        with h5py.File(results_filepath, 'a') as f:
            cm = f['compute_map'][:]
            f['compute_map'][:] = np.logical_or(mask, cm)

    if verbose:
        print('')

    vol_list = _build_equally_spaced_volume_list(
        spacing=np.array(target_size) - np.array(overlap), area_size=size[:3], n_volumes=1, transform_ratio=0
    )

    for idx, (xyz, volume, transform) in enumerate(vol_list):

        xyz = xyz.astype('int16')

        x = int(xyz[2] + size[2] / 2)
        y = int(xyz[1] + size[1] / 2)
        z = int(xyz[0] + size[0] / 2)

        if verbose:
            sys.stdout.write('\r' + 'x = {}; y = {}, z = {}'.format(x, y, z))

        if use_compute_map:
            with h5py.File(results_filepath, 'r') as f:
                if downsample_compute_map:
                    compute = not f['compute_map'][int(z / downsample_compute_map), int(y / downsample_compute_map), int(x / downsample_compute_map)]
                else:
                    compute = not f['compute_map'][z, y, x]
        else:
            compute = True

        if compute:

            # __________________________________
            # Load and transform the raw volumes

            # Load data
            im, _ = _load_data_with_padding(raw_channels[volume],
                                            xyz, target_size,
                                            auto_pad=False,
                                            auto_pad_z=False)
            im = im.astype('float32')

            check_thresh = np.ones(im.shape)
            check_thresh[im < thresh[0]] = 0
            check_thresh[im > thresh[1]] = 0
            if compute_empty_volumes or check_thresh.sum():

                # Scale everything for computation
                if scale != 1:
                    pass
                    # # im = np.swapaxes(im, 0, -1)
                    # # chs = []
                    # # for ch in im:
                    # #     chs.append(zoom(ch, scale))
                    # # im = np.swapaxes(np.array(chs), 0, -1)
                    #
                    # # pad image
                    # t_im = np.zeros(np.array(list(np.array(target_size) * scale) + [1]).astype(int), dtype=('uint8'))
                    # start = ((np.array(target_size) * scale - np.array(target_size)) / 2).astype(int)
                    # stop = (np.array(start) + np.array(target_size)).astype(int)
                    # start = list(start) + [0]
                    # stop = list(stop) + [1]
                    # t_im[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2], start[3]:stop[3]] = im
                    # im = t_im
                    #
                    # transform_matrix = np.array([[1/scale, 0, 0, 0],
                    #                         [0, 1/scale, 0, 0],
                    #                         [0, 0, 1/scale, 0],
                    #                         [0, 0, 0, 1]])
                    #
                    # # h, w = x.shape[img_x_axis], x.shape[img_y_axis]
                    # # transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
                    # fill_mode = 'nearest'
                    # cval = 0.
                    # w, h, d = im.shape[2], im.shape[1], im.shape[0]
                    # transform_matrix = transform_matrix_offset_center(transform_matrix, d, h, w)
                    # im = apply_transform(im, transform_matrix, 3,
                    #                      fill_mode=fill_mode, cval=cval)

                im = im.astype('float32') / 255
                im_in = np.moveaxis(im, -1, 0)[None, :]
                try:
                    imx = t.tensor(im_in, dtype=t.float32).cuda()
                except AssertionError:
                    print('Falling back to cpu...')
                    imx = t.tensor(im_in, dtype=t.float32).cpu()
                result = model(imx)
                result = result.cpu().numpy()
                result = np.moveaxis(result, 1, 4)[0, :]
                if squeeze_result:
                    result = result.squeeze()

                with h5py.File(results_filepath, 'a') as fr:
                    _write_result(
                        fr['data'], result, (xyz * scale).astype('int16'), ((np.array(target_size) - np.array(overlap)) * scale).astype('int16')
                    )
                    if use_compute_map:
                        if downsample_compute_map:
                            fr['compute_map'][int(z / downsample_compute_map), int(y / downsample_compute_map), int(x / downsample_compute_map)] = True
                        else:
                            fr['compute_map'][z, y, x] = True
                    # if scale != 1:
                    #     _write_result(
                    #         fr['scaled_input'], (im * 255).astype('uint8'), (xyz * scale).astype('int16'), (np.array(overlap) * scale).astype('int16')
                    #     )

            else:

                if verbose:
                    print(' skipped...')

        else:
            if verbose:
                print(' already computed...')


if __name__ == '__main__':

    def _dummy(vol):
        return vol

    f = h5py.File('/g/emcf/common/5792_Sars-Cov-2/exp_070420/FIB-SEM/alignments/20-04-23_S4_area2_Sam/amst_inv_clip.h5', mode='r')
    raw_channels = [[f['t00000/s00/0/cells'][:32, :500, :100]]]

    with h5py.File('/g/schwab/hennies/tmp/test_run_full_input.h5', mode='w') as f:
        f.create_dataset('data', data=raw_channels[0][0])

    predict_model_from_h5(
        model=_dummy,
        results_filepath='/g/schwab/hennies/tmp/test_run_full.h5',
        raw_channels=raw_channels,
        num_result_channels=1,
        target_size=(40, 40, 40),
        overlap=(20, 20, 20),
        scale=1.6,
        compute_empty_volumes=False,
        thresh=[16, 238],
        use_compute_map=True,
        mask=None
    )

    f.close()
