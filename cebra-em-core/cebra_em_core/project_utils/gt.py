
from cebra_em_core.project_utils.project import get_current_project_path
from cebra_em_core.project_utils.config import (
    get_config,
    get_config_filepath,
    add_to_config_json
)
from cebra_em_core.dataset.bdv_utils import bdv2pos

import re
import numpy as np


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
