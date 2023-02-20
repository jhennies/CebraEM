
import numpy as np
import pickle

from cebra_em.run_utils.tasks import load_data
from cebra_em_core.project_utils.project import get_current_project_path
from cebra_em_core.project_utils.config import get_config, absolute_path
from cebra_em.run_utils.run_specs import get_run_json
from pybdv.metadata import get_data_path, get_key
from cebra_em_core.dataset.bdv_utils import is_h5
from cebra_em_core.bioimageio.cebra_net import run_cebra_net


if __name__ == '__main__':

    print(f">>> STARTING: Membrane prediction for {snakemake.params['image_name']}[{snakemake.wildcards['idx']}]")

    # _______________________________________________________________________________
    # Retrieving settings

    project_path = get_current_project_path(None)
    run_json = get_run_json(project_path)

    verbose = run_json['verbose']

    try:
        mask_xml = get_config('mask', project_path)['xml_path']
    except KeyError:
        mask_xml = None

    dataset = snakemake.params['image_name']
    idx = int(snakemake.wildcards['idx'])

    config_ds = get_config(dataset, project_path)
    dep_datasets = config_ds['dep_datasets']
    target_resolution = config_ds['resolution']
    positions_fp = absolute_path(config_ds['positions'])
    halo = config_ds['halo']
    batch_shape = config_ds['batch_shape']

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

    if mask_xml is not None:
        config_mask = get_config('mask', project_path)
        dep_datasets.append('mask')
        dep_data_paths.append(get_data_path(mask_xml, return_absolute_path=True))
        dep_keys.append(get_key(is_h5(mask_xml), 0, 0, 0))
        dep_resolutions.append(config_mask['resolution'])
        dep_orders.append(0)

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
        print(f'input_data["raw"].shape = {input_data["raw"].shape}')

    # Compute membrane prediction
    output_data = run_cebra_net(input_data['raw'])

    # Save the result
    # TODO!

    # Write result file
    open(snakemake.output[0], 'w').close()
