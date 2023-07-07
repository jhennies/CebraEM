
import numpy as np
from cebra_em_core.dataset.data import crop_and_scale
from cebra_em_core.dataset.data import crop_zero_padding_3d


def load_data(
        dep_datasets,
        input_paths,
        input_keys,
        input_resolutions,
        interpolation_orders,
        target_resolution,
        position_halo,
        shape_halo,
        xcorr_on_raw=False,
        verbose=False
):

    output = {}
    for idx, input_layer in enumerate(dep_datasets):

        data = crop_and_scale(
            input_path=input_paths[idx],
            position=np.array(position_halo),
            internal_path=input_keys[idx],
            input_res=input_resolutions[idx],
            output_res=target_resolution,
            output_shape=shape_halo,
            order=interpolation_orders[idx],
            xcorr=False if input_layer != 'raw' else xcorr_on_raw,
            verbose=verbose
        )

        if list(data.shape) != list(shape_halo):
            # Sometimes rounding issues make the data a pixel larger than intended
            data = data[:shape_halo[0], :shape_halo[1], :shape_halo[2]]

        assert list(data.shape) == list(shape_halo), \
            f'data.shape = {data.shape} != shape_halo = {shape_halo}'

        output[input_layer] = data

    return output


def apply_normalization(
        raw,
        mask,
        mask_ids,
        raw_quantiles,
        relative_quantiles,
        verbose=False
):
    """

    :param raw:
    :param mask:
    :param mask_ids:
    :param raw_quantiles: Measured quantiles from the raw dataset, a dict of this structure:
        raw_quantiles = {
            [mask_id]: { [percentile]: [value] }
        }
        Note: The percentiles are here noted as p * 100
    :param relative_quantiles: Information of how to normalize in relative values
        relative_quantiles = {
            "quantiles": [0.1, 0.9],
            "anchors": [0.05, 0.95]
        }
    :param verbose:
    :return:
    """
    print('Applying normalization ... ')
    if verbose:
        print(f'raw_quantiles = {raw_quantiles}')
        print(f'relative_quantiles = {relative_quantiles}')

    assert raw.dtype == 'uint8'

    alow, ahigh = np.array(relative_quantiles['anchors']) * 255

    def normalize(pixels, alow, ahigh, qlow, qhigh):
        pixels = pixels.astype('float64')
        pixels -= qlow
        pixels /= qhigh - qlow
        pixels *= ahigh - alow
        pixels += alow
        pixels[pixels < 0] = 0
        pixels[pixels > 255] = 255
        pixels = np.round(pixels).astype('uint8')
        return pixels

    if mask is not None:

        for idx in np.unique(mask):
            if idx in mask_ids:

                qlow = raw_quantiles[str(idx)][str(int(relative_quantiles['quantiles'][0] * 100))]
                qhigh = raw_quantiles[str(idx)][str(int(relative_quantiles['quantiles'][1] * 100))]

                raw[mask == idx] = normalize(raw[mask == idx], alow, ahigh, qlow, qhigh)

    else:

        qlow = raw_quantiles['0'][str(int(relative_quantiles['quantiles'][0] * 100))]
        qhigh = raw_quantiles['0'][str(int(relative_quantiles['quantiles'][1] * 100))]

        raw = normalize(raw, alow, ahigh, qlow, qhigh)

    return raw


def compute_task_with_mask(func, vol, mask, mask_ids, halo=None, pad_result_vol=True, verbose=False):
    """

    :param func: The function running the task with this signature:
        def my_task_func(vol, mask=None):
            pass
    :param vol:
    :param mask:
    :param mask_ids:
    :param halo:
    :param pad_result_vol:
    :param verbose:
    :return:
    """

    vol = np.array(vol)

    def _get_bin_mask():
        if mask is None:
            if vol.ndim == 3:
                tm = np.ones(vol.shape, dtype=bool)
            elif vol.ndim == 4:
                tm = np.ones(vol[0].shape, dtype=bool)
            else:
                raise NotImplementedError('Only 3 and 4 dimensional arrays are implemented!')
        else:
            tm = np.zeros(mask.shape, dtype=bool)
            for idx in mask_ids:
                tm[mask == idx] = True
        return tm

    bin_mask = _get_bin_mask()

    if halo is not None:
        halo_mask = bin_mask[halo[0]: -halo[0],
                             halo[1]: -halo[1],
                             halo[2]: -halo[2]]
    else:
        halo_mask = bin_mask.copy()

    # Return if there is no data in the main ROI
    if not halo_mask.any() and pad_result_vol:  # FIXME
        if verbose:
            print(f'No data inside ROI, returning zeros ...')
        return np.zeros(vol.shape)
        # --------------------------------------

    # Simply compute if there is no background in the mask
    if bin_mask.min() > 0 and pad_result_vol:  # FIXME
        if verbose:
            print(f'Mask fully within data, computing normally ...')
        return func(vol)
        # --------------------------------------

    if verbose:
        print(f'Mask contains zeros, cropping to bounds ...')
    # Crop the bounding rect of the mask
    bounds = crop_zero_padding_3d(bin_mask, add_halo=halo)
    if np.array(vol).ndim == 3:
        vol_in = vol[bounds]
    elif np.array(vol).ndim == 4:
        vol_in = [v[bounds] for v in vol]
    else:
        raise RuntimeError('Only 3 or 4 dimensional arrays implemented!')

    # Compute the respective subarea
    res = func(vol_in, mask=bin_mask[bounds])

    print(f'pad_result_vol = {pad_result_vol}')
    if pad_result_vol:

        if verbose:
            print(f'res.shape = {res.shape}')
            print(f'np.unique(res) = {np.unique(res)}')

        if verbose:
            print(f'Padding result to original size ...')
        # Pad the result to match the input shape
        vol_out = np.zeros(vol.shape, res.dtype)
        vol_out[bounds] = res
        if verbose:
            print(f'vol_out.shape = {vol_out.shape}')
            print(f'np.unique(vol_out) = {np.unique(vol_out)}')

        if verbose:
            print(f'Removing everything outside the mask ...')
        # Remove everything outside the mask
        vol_out[np.logical_not(bin_mask)] = 0
        if verbose:
            print(f'np.unique(vol_out) = {np.unique(vol_out)}')

        return vol_out

    else:
        return dict(
            result=res, mask=bin_mask[bounds], bounds=bounds, shape=bin_mask.shape
        )
