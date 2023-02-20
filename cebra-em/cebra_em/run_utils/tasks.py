
import numpy as np
from cebra_em_core.misc.data import crop_and_scale


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
