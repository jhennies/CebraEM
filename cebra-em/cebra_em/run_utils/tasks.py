
import numpy as np
from cebra_em_core.dataset.data import crop_and_scale


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

    for idx in np.unique(mask):
        if idx in mask_ids:

            qlow = raw_quantiles[str(idx)][str(int(relative_quantiles['quantiles'][0] * 100))]
            qhigh = raw_quantiles[str(idx)][str(int(relative_quantiles['quantiles'][1] * 100))]

            pixels = raw[mask == idx]
            pixels = pixels.astype('float64')
            pixels -= qlow
            pixels /= qhigh - qlow
            pixels *= ahigh - alow
            pixels += alow
            pixels[pixels < 0] = 0
            pixels[pixels > 255] = 255
            pixels = np.round(pixels).astype('uint8')
            raw[mask == idx] = pixels

    return raw


