
import numpy as np
import pickle
import json
import os
from pybdv.metadata import get_data_path
from pybdv.util import open_file, get_key

# from pybdv.bdv_datasets import BdvDataset

from cebra_em_core.project_utils.config import get_config, absolute_path
from cebra_em_core.project_utils.project import get_current_project_path
from cebra_em_core.dataset.bdv_utils import is_h5
from cebra_em.run_utils.run_specs import get_run_json
from cebra_em.misc.bdv_io import vol_to_bdv


def apply_mapping(data, mapping, verbose=False):

    labels = np.unique(data)
    new = data.copy()

    for lbl in labels:
        if lbl > 0:
            if str(lbl) in mapping.keys():
                if verbose:
                    print(f'lbl = {lbl} ')
                    print(f'mapping[str(lbl)] = {mapping[str(lbl)]}')
                new[data == lbl] = mapping[str(lbl)][0]
            else:
                if verbose:
                    print(f'Label not found in mapping: {lbl}')

    return new


def load_data(data_path, data_key, position, shape):

    with open_file(data_path, mode='r') as f:
        data = f[data_key][
            position[0]: position[0] + shape[0],
            position[1]: position[1] + shape[1],
            position[2]: position[2] + shape[2]
        ]
    return data


# def save_data(data, data_path, position, shape):
#     bdv_ds = BdvDataset(
#         data_path,
#         timepoint=0,
#         setup_id=0,
#         downscale_mode='nearest',  # It's always a segmentation
#         n_threads=1,
#         verbose=True
#     )
#     bdv_ds[
#         position[0]: position[0] + shape[0],
#         position[1]: position[1] + shape[1],
#         position[2]: position[2] + shape[2]
#     ] = data


if __name__ == '__main__':

    image = snakemake.params['image_name']
    cube_idx = snakemake.wildcards['idx']

    print(f">>> STARTING: Apply mapping for {image}[{cube_idx}]")

    # _______________________________________________________________________________
    # Retrieving settings

    project_path = get_current_project_path(None)
    verbose = get_run_json(project_path=project_path)['verbose']
    beta = get_run_json(project_path=project_path)['misc']['beta']

    # First input is the mapping
    mapp_filepath = snakemake.input[0]
    output = snakemake.output[0]

    # Get the config
    config_seg = get_config(image, project_path=project_path)
    positions_fp = absolute_path(config_seg['positions'], project_path=project_path)
    # data_xml_path = absolute_path(config_seg['xml_path'])
    # stitched_data_xml = absolute_path(config_seg['stitched_dataset']['xml_path'], project_path=project_path)
    # stitched_data_path = get_data_path(stitched_data_xml, return_absolute_path=True)

    stitched_img_xml_rel_path = config_seg['segmentations'][f'{image}_b{str.replace(str(beta), ".", "_")}']['xml_path_stitched']
    stitched_img_data_path = get_data_path(
        absolute_path(stitched_img_xml_rel_path, project_path=project_path),
        return_absolute_path=True
    )
    img_xml_rel_path = config_seg['segmentations'][f'{image}_b{str.replace(str(beta), ".", "_")}']['xml_path']
    img_xml_abs_path = absolute_path(img_xml_rel_path, project_path=project_path)

    # Get shape and position
    shp = config_seg['batch_shape']
    with open(positions_fp, mode='rb') as f:
        positions = pickle.load(f)
        if verbose:
            print(positions)
            print(f'Loading position id {cube_idx}')
        pos = positions[int(cube_idx)]
        if verbose:
            print(f'Position = {pos}')

    # _______________________________________________________________________________
    # Process the data

    # Load data
    data_path = get_data_path(img_xml_abs_path, return_absolute_path=True)
    data_key = get_key(is_h5(img_xml_abs_path), 0, 0, 0)
    data = load_data(data_path, data_key, pos, shp)
    shp = data.shape
    if verbose:
        print(f'labels = {np.unique(data)}')
        print(f'mapp_filepath = {mapp_filepath}')

    # with open(mapp_filepath, mode='rb') as f:
    #     mapp = pickle.load(f)
    with open(mapp_filepath, mode='r') as f:
        mapp = json.load(f)

    if verbose:
        print(f'mapping = {mapp}')

    # Apply mapping
    data = apply_mapping(data, mapp, verbose=verbose)

    # Save result
    if verbose:
        print('Saving data:')
        print(f'labels = {np.unique(data)}')
        print(f'shape = {data.shape}')
        print(f'shp = {shp}')
        print(f'pos = {pos}')
    # save_data(data, stitched_img_data_path, pos, shp)
    # _update_table(table_path, bdv_ds.get_max_id())
    vol_to_bdv(
        data,
        dataset_path=stitched_img_data_path,
        position=pos,
        downscale_mode='nearest',
        halo=None,
        background_value=0,
        unique=False,
        update_max_id=False,
        cast_type=None,
        block_description=dict(
            path=project_path,
            idx=int(cube_idx),
            name=image
        ),
        verbose=verbose
    )

    # _______________________________________________________________________________
    # Write result file
    open(output, 'w').close()

    print(f"<<< DONE: Apply mapping for {image}[{cube_idx}]")
