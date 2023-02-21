
import numpy as np
import pickle

from cebra_em.run_utils.tasks import load_data
from cebra_em_core.project_utils.project import get_current_project_path
from cebra_em_core.project_utils.config import get_config, absolute_path
from cebra_em.run_utils.run_specs import get_run_json
from pybdv.metadata import get_data_path, get_key
from cebra_em_core.dataset.bdv_utils import is_h5
from cebra_em_core.bioimageio.cebra_net import run_cebra_net
from cebra_em.misc.bdv_io import vol_to_bdv
from cebra_em_core.segmentation.supervoxels import watershed_dt_with_probs


def out_of_mask(input_dict, mask_ids):
    mask = input_dict['mask'] if 'mask' in input_dict else None

    # Do not compute if completely outside the mask
    if mask_ids is not None:
        assert mask is not None
        tmask = np.zeros(mask.shape, dtype=bool)
        for idx in mask_ids:
            tmask[mask == idx] = True
        if not tmask.any():
            return True
        del mask, tmask
    return False


def run_membrane_prediction(
        input_dict,
        mask_ids=None
):
    assert 'raw' in input_dict
    if 'mask' in input_dict:
        assert len(input_dict) == 2
    else:
        assert len(input_dict) == 1

    raw = input_dict['raw']

    # Do not compute if completely outside the mask
    if out_of_mask(input_dict, mask_ids):
        return np.zeros(raw.shape)
    else:
        return run_cebra_net(raw).squeeze()


def run_supervoxels(
        input_dict,
        sv_kwargs,
        mask_ids=None
):
    assert 'membrane_prediction' in input_dict
    if 'mask' in input_dict:
        assert len(input_dict) == 2
    else:
        assert len(input_dict) == 1

    mem = input_dict['membrane_prediction']

    # Do not compute if completely outside the mask
    if out_of_mask(input_dict, mask_ids):
        return np.zeros(mem.shape)
    else:
        return watershed_dt_with_probs(
            mem,
            **sv_kwargs,
            verbose=verbose
        )


if __name__ == '__main__':

    print(f">>> STARTING: Run for {snakemake.params['image_name']}[{snakemake.wildcards['idx']}]")

    # _______________________________________________________________________________
    # Retrieving settings

    project_path = get_current_project_path(None)
    run_json = get_run_json(project_path)

    verbose = run_json['verbose']

    dataset = snakemake.params['image_name']
    idx = int(snakemake.wildcards['idx'])

    # Config and run settings of this dataset
    config_ds = get_config(dataset, project_path)
    ds_xml_path = absolute_path(config_ds['xml_path'], project_path=project_path)
    ds_path = get_data_path(ds_xml_path, return_absolute_path=True)
    dep_datasets = config_ds['dep_datasets']
    target_resolution = config_ds['resolution']
    positions_fp = absolute_path(config_ds['positions'])
    halo = config_ds['halo']
    batch_shape = config_ds['batch_shape']
    data_writing = config_ds['data_writing']

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

    if verbose:
        print(f'input_data.keys() = {input_data.keys()}')

    # _______________________________________________________________________________
    # Run the task

    if dataset == 'membrane_prediction':
        output_data = run_membrane_prediction(input_data, mask_ids=mask_ids)
    elif dataset == 'supervoxels':
        sv_kwargs = config_ds['sv_kwargs']
        output_data = run_supervoxels(input_data, sv_kwargs, mask_ids=mask_ids)
    else:
        raise RuntimeError(f'Invalid dataset: {dataset}')

    # _______________________________________________________________________________
    # Save the result
    positions_fp = absolute_path(config_ds['positions'], project_path=project_path)
    with open(positions_fp, 'rb') as f:
        pos = pickle.load(f)[idx]

    vol_to_bdv(
        output_data,
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



