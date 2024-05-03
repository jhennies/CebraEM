
import numpy as np
import pickle
import os
import json

from cebra_em.run_utils.tasks import load_data, apply_normalization
from cebra_em_core.project_utils.project import get_current_project_path
from cebra_em_core.project_utils.config import get_config, absolute_path
from cebra_em.run_utils.run_specs import get_run_json
from pybdv.metadata import get_data_path, get_key
from cebra_em_core.dataset.bdv_utils import is_h5
from cebra_em_core.bioimageio.cebra_net import run_cebra_net
from cebra_em.misc.bdv_io import vol_to_bdv
from cebra_em_core.segmentation.supervoxels import watershed_dt_with_probs
from cebra_em.run_utils.tasks import compute_task_with_mask


def run_membrane_prediction(
        input_dict,
        mask_ids=None,
        halo=None,
        verbose=False
):
    assert 'raw' in input_dict
    if 'mask' in input_dict:
        assert len(input_dict) == 2
    else:
        assert len(input_dict) == 1

    raw = input_dict['raw']

    def _run_cebra_net(vol, mask=None):
        return run_cebra_net(vol).squeeze()
        # # Use this for debugging:
        # return np.ones(vol.shape) * 255

    if 'mask' in input_dict:
        if verbose:
            print(f'Computing with mask ...')
        return compute_task_with_mask(_run_cebra_net, raw, input_dict['mask'], mask_ids=mask_ids, halo=halo, verbose=verbose)
    else:
        if verbose:
            print(f'Computing without mask ...')
        return _run_cebra_net(raw)


def run_supervoxels(
        input_dict,
        sv_kwargs,
        mask_ids=None,
        halo=None,
        tile=True,
        verbose=False
):

    # TODO check if a mask can be applied to computation such that exhaustive memory hungry steps can be avoided
    assert 'membrane_prediction' in input_dict
    if 'mask' in input_dict:
        assert len(input_dict) == 2
    else:
        assert len(input_dict) == 1

    mem = input_dict['membrane_prediction']

    def run_sv(vol, mask=None):
        return watershed_dt_with_probs(vol, **sv_kwargs, verbose=verbose)
        # # Use this for debugging:
        # return np.ones(vol.shape)

    if tile:

        shp = np.array(mem.shape)
        hl = np.array(halo)
        pos = []
        pos_t = []
        for z in range(2):
            for y in range(2):
                for x in range(2):
                    pos.append(np.s_[
                        int(z * (shp[0]/2 - hl[0])): int(z * (shp[0]/2 - hl[0]) + shp[0]/2 + hl[0]),
                        int(y * (shp[1]/2 - hl[1])): int(y * (shp[1]/2 - hl[1]) + shp[1]/2 + hl[1]),
                        int(x * (shp[2]/2 - hl[2])): int(x * (shp[2]/2 - hl[2]) + shp[2]/2 + hl[2])
                    ])
                    pos_t.append(np.s_[
                        int((1-z) * hl[0] + z * shp[0]/2): int((1-z) * hl[0] + z * shp[0]/2 + shp[0]/2 - hl[0]),
                        int((1-y) * hl[1] + y * shp[1]/2): int((1-y) * hl[1] + y * shp[1]/2 + shp[1]/2 - hl[1]),
                        int((1-x) * hl[2] + x * shp[2]/2): int((1-x) * hl[2] + x * shp[2]/2 + shp[2]/2 - hl[2])
                    ])

        result_vol = np.zeros(mem.shape, dtype='float32')
        if 'mask' in input_dict:

            mask = input_dict['mask']

            max_val = 0
            for idx, p in enumerate(pos):
                result_vol[pos_t[idx]] = compute_task_with_mask(run_sv, mem[p], mask[p], mask_ids, halo=halo)[
                                            hl[0]: -hl[0],
                                            hl[1]: -hl[1],
                                            hl[2]: -hl[2],
                                         ] + max_val
                max_val = result_vol.max()
            return result_vol

        else:
            max_val = 0
            for idx, p in enumerate(pos):
                result_vol[pos_t[idx]] = run_sv(mem[p])[
                                             hl[0]: -hl[0],
                                             hl[1]: -hl[1],
                                             hl[2]: -hl[2],
                                         ] + max_val
                max_val = result_vol.max()
            return result_vol

    if 'mask' in input_dict:

        mask = input_dict['mask']
        return compute_task_with_mask(run_sv, mem, mask, mask_ids=mask_ids, halo=halo)
    else:
        return run_sv(mem)


if __name__ == '__main__':

    print(f">>> STARTING: Run for {snakemake.params['image_name']}[{snakemake.wildcards['idx']}]")

    # _______________________________________________________________________________
    # Retrieving settings

    project_path = get_current_project_path(None)
    run_json = get_run_json(project_path)

    verbose = run_json['verbose']
    debug = run_json['debug']

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

    relative_quantiles = config_ds['quantile_norm'] if 'quantile_norm' in config_ds else None
    if relative_quantiles is not None:
        raw_quantiles_fp = snakemake.input[-1]
        assert raw_quantiles_fp == os.path.join(project_path, 'snk_wf', 'raw_quantiles.json'), \
            f"{raw_quantiles_fp} != {os.path.join(project_path, 'snk_wf', 'raw_quantiles.json')}"
        with open(raw_quantiles_fp, mode='r') as f:
            raw_quantiles = json.load(f)
    else:
        raw_quantiles = None

    invert = config_raw['invert'] if 'invert' in config_raw else None

    # _______________________________________________________________________________
    # Write result file
    # NOTE: doing this before computing anything. Like this, I can get all this information in case it errors out
    positions_fp = absolute_path(config_ds['positions'], project_path=project_path)
    with open(positions_fp, 'rb') as f:
        pos = pickle.load(f)[idx]

    with open(snakemake.output[0], 'w') as f:
        json.dump(
            dict(
                dataset_path=ds_path,
                position=pos.tolist(),
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
                )
            ),
            f, indent=2
        )

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
    # Pre-processing

    if relative_quantiles is not None:
        input_data['raw'] = apply_normalization(
            input_data['raw'],
            input_data['mask'] if 'mask' in input_data else None,
            mask_ids,
            raw_quantiles,
            relative_quantiles,
            verbose=verbose
        )
    if invert is not None and 'raw' in input_data:
        if invert:
            assert input_data['raw'].dtype == 'uint8', 'Data inversion only implemented for 8bit data!'
            input_data['raw'] = 255 - input_data['raw']

    # _______________________________________________________________________________
    # Run the task

    if dataset == 'membrane_prediction':
        if debug:
            from elf.io import open_file
            filename = os.path.join(os.path.splitext(snakemake.output[0])[0] + '.h5')
            with open_file(filename, mode='w') as f:
                f.create_dataset('data', data=input_data['raw'], compression='gzip')
        output_data = run_membrane_prediction(input_data, mask_ids=mask_ids, halo=halo, verbose=verbose)
    elif dataset == 'supervoxels':
        sv_kwargs = config_ds['sv_kwargs']
        output_data = run_supervoxels(input_data, sv_kwargs, mask_ids=mask_ids, halo=halo, verbose=verbose)
    else:
        raise RuntimeError(f'Invalid dataset: {dataset}')

    # _______________________________________________________________________________
    # Save the result

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

