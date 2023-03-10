
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

from cebra_em_core.segmentation.multicut import multicut_from_predicted


# def predict_segmentation(
#         input_dict, seg_kwargs, mask_ids=None, halo=None, verbose=False
# ):
#
#     if verbose:
#         print(f'keys = {input_dict.keys()}')
#     assert 'rf_filepath' in input_dict
#     assert 'nrf_filepath' in input_dict
#     assert 'raw' in input_dict
#     assert 'supervoxels' in input_dict
#     assert 'membrane_prediction' in input_dict
#     mask = input_dict['mask'] if 'mask' in input_dict else None
#     vol = [
#         input_dict['raw'],
#         input_dict['membrane_prediction'],
#         input_dict['supervoxels']
#     ]
#     rf_filepath = input_dict['rf_filepath']
#     nrf_filepath = input_dict['nrf_filepath']
#
#     if verbose:
#         print(f"raw.dtype = {input_dict['raw'].dtype}")
#         print(f"supervoxels.dtype = {input_dict['supervoxels'].dtype}")
#         print(f"membrane_prediction.dtype = {input_dict['membrane_prediction'].dtype}")
#
#     def run_predict_mc_wf_classifiers(vol, mask=None):
#         return predict_mc_wf_classifiers(vol, rf_filepath, nrf_filepath, mask=mask, **seg_kwargs, verbose=verbose)
#         # # Use this for debugging:
#         # return np.ones(vol.shape)
#
#     out_dict = compute_task_with_mask(
#         run_predict_mc_wf_classifiers, vol, mask,
#         mask_ids=mask_ids, halo=halo, pad_result_vol=False,
#         verbose=verbose
#     )
#
#     mc_out, bounds, mask = out_dict['result'], out_dict['bounds'], out_dict['mask']
#
#     return dict(mc_out=mc_out, bounds=bounds, mask=mask)


if __name__ == '__main__':

    print(f">>> STARTING: Run for {snakemake.params['image_name']}[{snakemake.wildcards['idx']}]")

    # _______________________________________________________________________________
    # Retrieving settings

    beta = float(snakemake.wildcards['beta'])

    project_path = get_current_project_path(None)
    run_json = get_run_json(project_path)

    verbose = run_json['verbose']

    base_dataset = snakemake.params['image_name']
    dataset = f"{snakemake.params['image_name']}_b{str.replace(str(beta), '.', '_')}"
    idx = int(snakemake.wildcards['idx'])

    print(f'dataset = {dataset}')
    print(f'idx = {idx}')

    # # Config and run settings of this dataset
    config_ds = get_config(base_dataset, project_path)['segmentations'][dataset]
    ds_xml_path = absolute_path(config_ds['xml_path'], project_path=project_path)
    ds_path = get_data_path(ds_xml_path, return_absolute_path=True)
    target_resolution = config_ds['resolution']
    data_writing = config_ds['data_writing']

    config_base_ds = get_config(base_dataset, project_path)
    positions_fp = absolute_path(config_base_ds['positions'])
    halo = config_base_ds['halo']
    batch_shape = config_base_ds['batch_shape']

    # Add the mask to the dependencies
    try:
        mask_xml = get_config('mask', project_path)['xml_path']
    except KeyError:
        mask_xml = None

    if mask_xml is not None:
        config_mask = get_config('mask', project_path)

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

    # # _______________________________________________________________________________
    # # Retrieve the input

    with open(snakemake.input[0], mode='rb') as f:
        inp_data = pickle.load(f)

    if verbose:
        print(f'beta = {beta}')

    # _______________________________________________________________________________
    # Run the task

    # seg_kwargs = {}  # config_ds['mc_args']
    # output_data = predict_segmentation(input_data, seg_kwargs, mask_ids=mask_ids, halo=halo, verbose=verbose)
    seg = multicut_from_predicted(inp_data, beta, verbose=verbose)

    # _______________________________________________________________________________
    # Save the result
    positions_fp = absolute_path(config_base_ds['positions'], project_path=project_path)
    with open(positions_fp, 'rb') as f:
        pos = pickle.load(f)[idx]

    vol_to_bdv(
        seg,
        dataset_path=ds_path,
        position=pos,
        downscale_mode=data_writing['downscale_mode'],
        halo=halo,
        background_value=data_writing['background_value'],
        unique=data_writing['unique_labels'],
        update_max_id=data_writing['unique_labels'],
        cast_type=data_writing['dtype'] if 'dtype' in data_writing.keys() else None,
        block_description=dict(
            path=project_path,
            idx=idx,
            name=dataset
        ),
        verbose=verbose
    )

    # _______________________________________________________________________________
    # Write result file
    open(snakemake.output[0], 'w').close()
