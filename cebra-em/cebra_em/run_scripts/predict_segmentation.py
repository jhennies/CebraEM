
import numpy as np
import pickle
import os
import json
import pickle

from cebra_em.run_utils.tasks import load_data, apply_normalization
from cebra_em_core.project_utils.project import get_current_project_path
from cebra_em_core.project_utils.config import get_config, absolute_path
from cebra_em.run_utils.run_specs import get_run_json
from pybdv.metadata import get_data_path, get_key
from cebra_em_core.dataset.bdv_utils import is_h5
from cebra_em_core.bioimageio.cebra_net import run_cebra_net
from cebra_em.misc.bdv_io import vol_to_bdv
from cebra_em_core.segmentation.supervoxels import watershed_dt_with_probs
from cebra_em_core.dataset.data import crop_zero_padding_3d
from cebra_em.run_utils.tasks import compute_task_with_mask
from cebra_em_core.segmentation.multicut import predict_mc_wf_classifiers


def predict_segmentation(
        input_dict, seg_kwargs, mask_ids=None, halo=None, verbose=False
):

    if verbose:
        print(f'keys = {input_dict.keys()}')
    assert 'rf_filepath' in input_dict
    assert 'nrf_filepath' in input_dict
    assert 'raw' in input_dict
    assert 'supervoxels' in input_dict
    assert 'membrane_prediction' in input_dict
    mask = input_dict['mask'] if 'mask' in input_dict else None
    vol = [
        input_dict['raw'],
        input_dict['membrane_prediction'],
        input_dict['supervoxels']
    ]
    rf_filepath = input_dict['rf_filepath']
    nrf_filepath = input_dict['nrf_filepath']

    if verbose:
        print(f"raw.dtype = {input_dict['raw'].dtype}")
        print(f"supervoxels.dtype = {input_dict['supervoxels'].dtype}")
        print(f"membrane_prediction.dtype = {input_dict['membrane_prediction'].dtype}")

    def run_predict_mc_wf_classifiers(vol, mask=None):
        return predict_mc_wf_classifiers(vol, rf_filepath, nrf_filepath, mask=mask, **seg_kwargs, verbose=verbose)
        # # Use this for debugging:
        # return {'watershed': np.ones(vol.shape), 'edge_probs': [], 'node_probs': [], 'edge_sizes': []}

    out_dict = compute_task_with_mask(
        run_predict_mc_wf_classifiers, vol, mask,
        mask_ids=mask_ids, halo=halo, pad_result_vol=False,
        verbose=verbose
    )

    print(f'out_dict.keys() = {out_dict.keys()}')
    return out_dict

    # mc_out, bounds, mask = out_dict['result'], out_dict['bounds'], out_dict['mask']
    #
    # return dict(mc_out=mc_out, bounds=bounds, mask=mask)


if __name__ == '__main__':

    print(f">>> STARTING: Run for {snakemake.params['image_name']}[{snakemake.wildcards['idx']}]")

    # _______________________________________________________________________________
    # Retrieving settings

    project_path = get_current_project_path(None)
    run_json = get_run_json(project_path)

    verbose = run_json['verbose']

    dataset = snakemake.params['image_name']
    idx = int(snakemake.wildcards['idx'])

    # # Config and run settings of this dataset
    config_ds = get_config(dataset, project_path)
    dep_datasets = config_ds['dep_datasets']
    target_resolution = config_ds['resolution']
    positions_fp = absolute_path(config_ds['positions'])
    halo = config_ds['halo']
    batch_shape = config_ds['batch_shape']

    # Fetch the settings of the dependencies
    config_raw = get_config('raw', project_path)
    xcorr = config_raw['xcorr']

    dep_data_paths = []
    dep_keys = []
    dep_resolutions = []
    dep_orders = []
    for dep_ds in dep_datasets:
        config_dep = get_config(dep_ds, project_path)
        dep_xml_path = absolute_path(config_dep['xml_path'])
        dep_data_paths.append(get_data_path(dep_xml_path, return_absolute_path=True))
        dep_resolutions.append(config_dep['resolution'])
        dep_keys.append(get_key(is_h5(dep_xml_path), 0, 0, 0))
        if 'data_writing' in config_dep.keys():
            if config_dep['data_writing']['type'] == 'segmentation':
                dep_orders.append(0)
            elif config_dep['data_writing']['type'] == 'image':
                dep_orders.append(1)
            else:
                raise ValueError(f"Invalid type: {config_dep['data_writing']['type']}")
        else:
            dep_orders.append(1)  # The raw data doesn't have a data_writing attribute

    # Add the mask to the dependencies
    try:
        mask_xml = get_config('mask', project_path)['xml_path']
    except KeyError:
        mask_xml = None

    if mask_xml is not None:
        config_mask = get_config('mask', project_path)
        dep_datasets.append('mask')
        dep_data_paths.append(get_data_path(mask_xml, return_absolute_path=True))
        dep_keys.append(get_key(is_h5(mask_xml), 0, 0, 0))
        dep_resolutions.append(config_mask['resolution'])
        dep_orders.append(0)

        mask_ids = config_mask['args']['ids']
    else:
        mask_ids = None

    # Position and halo
    with open(positions_fp, 'rb') as f:
        position = pickle.load(f)[idx]

    if halo is not None:
        position_halo = np.array(position) - np.array(halo)
        shape_halo = np.array(batch_shape) + 2 * np.array(halo)
    else:
        position_halo = np.array(position)
        shape_halo = np.array(batch_shape)

    # _______________________________________________________________________________
    # Retrieve the input
    input_data = load_data(
        dep_datasets,
        dep_data_paths,
        dep_keys,
        dep_resolutions,
        dep_orders,
        target_resolution,
        position_halo,
        shape_halo,
        xcorr_on_raw=xcorr,
        verbose=verbose
    )

    # Adding the random forest models to the input dict
    input_data['rf_filepath'] = snakemake.input[0]
    input_data['nrf_filepath'] = snakemake.input[1]

    if verbose:
        print(f'input_data.keys() = {input_data.keys()}')

    # _______________________________________________________________________________
    # Run the task

    seg_kwargs = {}  # config_ds['mc_args']
    output_data = predict_segmentation(input_data, seg_kwargs, mask_ids=mask_ids, halo=halo, verbose=verbose)

    # _______________________________________________________________________________
    # Write result file
    with open(snakemake.output[0], 'wb') as f:
        pickle.dump(output_data, f)

