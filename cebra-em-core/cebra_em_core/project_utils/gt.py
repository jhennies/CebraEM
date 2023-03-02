
from cebra_em_core.project_utils.project import get_current_project_path
from cebra_em_core.project_utils.config import (
    get_config,
    get_config_filepath,
    add_to_config_json,
    absolute_path
)
from cebra_em_core.dataset.bdv_utils import bdv2pos
from cebra_em_core.dataset.data import crop_and_scale

import re
import numpy as np
from pybdv.metadata import get_data_path
from pybdv.util import open_file, get_key


def id2str(cube_id):
    return 'gt{:03d}'.format(cube_id)


def init_gt_cube(
        project_path=None,
        shape=(256, 256, 256),
        position=None,
        bdv_position=None,
        no_padding=False,
        val=False,
        verbose=False
):

    name = 'val' if val else 'gt'

    config_main_fp = get_config_filepath('main', project_path=project_path)
    # Add an entry to the main config
    add_to_config_json(
        config_main_fp,
        {
            f'{name}_path': '{project_path}' + f'{name}',
            'configs': {
                name: '{project_path}' + f'config/config_{name}.json'
            }
        }
    )

    # Add an entry to config/config_gt.json
    try:
        gt_config = get_config(name, project_path=project_path)
        if verbose:
            print(gt_config)
        if len(gt_config.keys()) == 0:
            print(f'No existing {name} cubes found.')
            cube_id = 0
        else:
            int_cube_ids = [int(re.findall('\d+', cube)[0]) for cube in gt_config.keys()]
            print(f'Already existing {name} cubes: {int_cube_ids}')
            cube_id = int(np.max(int_cube_ids) + 1)
    except FileNotFoundError:
        print(f'No {name} config found, adding from scratch.')
        cube_id = 0

    print(f'New cube id: {cube_id}')

    if bdv_position is not None:
        assert position is None, 'Supply either bdv_position or position, not both!'
        pos = bdv2pos(bdv_position, resolution=get_config('supervoxels', project_path)['resolution'], verbose=verbose)
    else:
        assert position is not None, 'No position supplied, use either position or bdv_position!'
        pos = position

    # Convert the position to the top-left-front corner
    pos = np.array(pos)
    pos_center = pos.copy()
    pos = (pos - np.array(shape) / 2).astype(int)

    add_to_config_json(
        get_config_filepath(name, project_path=project_path),
        {
            id2str(cube_id): {
                'id': cube_id,
                'status': 'pending',  # This makes it a priority job and triggers the run
                'position': pos,
                'position_center': pos_center,  # This is what the user originally selected, not actually used though
                'shape': shape,
                'no_padding': no_padding
            }
        }
    )


def extract_gt(
        cube_id,
        raw_fp, mem_fp, sv_fp,
        val=False, project_path=None, verbose=False
):

    name = 'val' if val else 'gt'

    config_sv = get_config('supervoxels', project_path=project_path)
    config_raw = get_config('raw', project_path=project_path)
    config_mem = get_config('membrane_prediction', project_path=project_path)
    config_gt = get_config(name, project_path=project_path)

    # Get the supervoxels resolution
    output_res = config_sv['resolution']

    # Get cube position and shape
    position = config_gt[cube_id]['position']
    shape = config_gt[cube_id]['shape']

    # Extracting raw data (add 128 px halo for more context)
    input_res = config_raw['resolution']
    xcorr = config_raw['xcorr']
    input_path = get_data_path(config_raw['xml_path'], True)
    input_key = get_key(False, 0, 0, 0)
    if config_gt[cube_id]['no_padding']:
        raw_pos = np.array(position)
        raw_shp = np.array(shape)
    else:
        raw_pos = np.array(position) - np.array((128,) * 3)
        raw_shp = np.array(shape) + 2 * np.array((128,) * 3)
    if verbose:
        print('---')
        print(f'input_path = {input_path}')
        print(f'raw_pos = {raw_pos}')
        print(f'input_key = {input_key}')
        print(f'input_res = {input_res}')
        print(f'output_res = {output_res}')
        print(f'raw_shp = {raw_shp}')
        print('---')
    raw = crop_and_scale(
        input_path=input_path,
        position=raw_pos,
        internal_path=input_key,
        input_res=input_res,
        output_res=output_res,
        output_shape=raw_shp,
        xcorr=xcorr,
        verbose=verbose
    )

    with open_file(raw_fp, 'w') as f:
        f.create_dataset('data', data=raw, compression='gzip')

    # Extracting membrane prediction
    input_res = config_mem['resolution']
    input_path = get_data_path(absolute_path(config_mem['xml_path'], project_path=project_path), True)
    input_key = get_key(False, 0, 0, 0)
    if verbose:
        print('---')
        print(f'input_path = {input_path}')
        print(f'position = {position}')
        print(f'input_key = {input_key}')
        print(f'input_res = {input_res}')
        print(f'output_res = {output_res}')
        print(f'shape = {shape}')
        print('---')
    mem = crop_and_scale(
        input_path=input_path,
        position=position,
        internal_path=input_key,
        input_res=input_res,
        output_res=output_res,
        output_shape=shape,
        verbose=verbose
    )

    with open_file(mem_fp, 'w') as f:
        f.create_dataset('data', data=mem, compression='gzip')

    # Extracting supervoxels
    input_res = config_sv['resolution']
    input_path = get_data_path(absolute_path(config_sv['xml_path'], project_path=project_path), True)
    input_key = get_key(False, 0, 0, 0)
    sv = crop_and_scale(
        input_path=input_path,
        position=position,
        internal_path=input_key,
        input_res=input_res,
        output_res=output_res,
        output_shape=shape,
        verbose=verbose
    )

    with open_file(sv_fp, 'w') as f:
        f.create_dataset('data', data=sv, compression='gzip')

