
from cebra_em_core.project_utils.project import get_current_project_path
from cebra_em_core.project_utils.config import (
    get_config,
    get_config_filepath,
    add_to_config_json,
    absolute_path,
)
from cebra_em_core.dataset.bdv_utils import bdv2pos
from cebra_em_core.dataset.data import crop_and_scale

import re
import numpy as np
from pybdv.metadata import get_data_path
from pybdv.util import open_file, get_key
import os
from glob import glob


def id2str(cube_id):
    return 'gt{:03d}'.format(cube_id)


def init_gt_cube(
        project_path=None,
        shape=(256, 256, 256),
        position=None,
        bdv_position=None,
        no_padding=False,
        verbose=False
):

    name = 'gt'

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
        project_path=None, verbose=False
):

    name = 'gt'

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

    # Update gt config:  set 'status' to 'ready'
    add_to_config_json(
        get_config_filepath('gt', project_path=project_path),
        {
            cube_id: {
                'status': 'ready'
            }
        }
    )


def _validate_inputs(cube_id, organelle_id, image_id, project_path=None):

    config_gt = get_config('gt', project_path=project_path)
    config_main = get_config('main', project_path=project_path)

    cube_id = id2str(cube_id)
    assert cube_id in config_gt.keys(), 'Specified cube_id does not exist!'

    annotation_folder = absolute_path(config_main['gt_path'], project_path=project_path)
    available_annotations = [os.path.split(x)[1] for x in glob(os.path.join(annotation_folder, cube_id, '*.h5'))]

    if f'{organelle_id}.h5' not in available_annotations:
        print(f'The specified annotation "{organelle_id}" in cube "{cube_id}" is not available.')

    assert image_id in config_main['configs'].keys(), \
        'Specified dataset does not exist!'


def generate_cube_to_image_link(
        organelle, image_name,
        cube_config_entry,
        val=False,
        verbose=False
):

    def _find_existing():
        try:
            idx = [x['image'] for x in cube_config_entry['links']].index(image_name)
            # The link exists and can be replaced
            return idx
        except ValueError:
            # The cube is not linked to the dataset, link can be appended
            return None

    if 'links' in cube_config_entry.keys():
        existing_link = _find_existing()
        links = cube_config_entry['links']
        if existing_link is None:
            links.append(
                {
                    'image': image_name,
                    'organelle': organelle,
                    'val': val
                }
            )
        else:
            print(f'The link already exists and will be updated.')
            links[existing_link] = {
                'image': image_name,
                'organelle': organelle,
                'val': val
            }
    else:
        links = [
            {
                'image': image_name,
                'organelle': organelle,
                'val': val
            }
        ]
    cube_config_entry['links'] = links

    return cube_config_entry


def link_gt_cubes(
        cube_ids, organelle, image_name,
        val=False,
        project_path=None,
        verbose=False
):

    config_gt_fp = get_config_filepath('gt', project_path=project_path)
    config_gt = get_config('gt', project_path=project_path)

    # Check all cube_ids and make sure that the respective entries all exist
    for cube_id in cube_ids:
        assert id2str(cube_id) in config_gt.keys(), f'This cube does not have a config entry: {cube_id}'

    # Link the cubes
    for cube_id in cube_ids:

        print(f'Linking cube {cube_id} ... ')

        _validate_inputs(cube_id, organelle, image_name, project_path=project_path)

        # Extend the config entry for the respective cube
        cube_config_entry = generate_cube_to_image_link(
            organelle, image_name,
            config_gt[id2str(cube_id)],
            val=val, verbose=verbose
        )

        # Update the config
        add_to_config_json(
            config_gt_fp,
            {id2str(cube_id): cube_config_entry}
        )

        print(f'Cube {cube_id} linked successfully to {image_name} :-)')


def get_associated_gt_cubes(image, project_path=None):

    config_gt = get_config('gt', project_path=project_path)

    trains = {}
    vals = {}

    for cube_name, cube_info in config_gt.items():
        cube_id = cube_info['id']
        if 'links' in cube_info.keys():
            for link in cube_info['links']:
                if link['image'] == image:
                    if not link['val']:
                        trains[cube_name] = {
                            'organelle': link['organelle'],
                            'cube_id': cube_id,
                            'annotated': os.path.exists(
                                os.path.join(
                                    absolute_path(get_config('main', project_path)['gt_path'], project_path),
                                    id2str(cube_id), f'{link["organelle"]}.h5'
                                )
                            )
                        }
                    else:
                        vals[cube_name] = {
                            'organelle': link['organelle'],
                            'cube_id': cube_id,
                            'annotated': os.path.exists(
                                os.path.join(
                                    absolute_path(get_config('main', project_path)['gt_path'], project_path),
                                    id2str(cube_id), f'{link["organelle"]}.h5'
                                )
                            )
                        }

    return trains, vals


def log_gt_cube(cube_id, status, position, shape, links=None, project_path=None):

    print('')
    print('____________________________________________________________________________________')
    print(f'Name = {id2str(cube_id)}, ID = {cube_id}')
    print('')
    print('  Position: {:.3f}, {:.3f}, {:.3f} \tShape: {}, {}, {}'.format(*position, *shape))
    print(f'  Data extraction status: {status}')
    print('')
    if links is None or len(links) == 0:
        print('  Not linked to any datasets')
    else:
        print('  LINKS \tType  \tLayer \tDataset \tAnnotated')
        print('')
        for link_id, link in enumerate(links):
            if not link['val']:
                gt_type = 'TRAIN'
            else:
                gt_type = 'VAL  '
            is_annotated = os.path.exists(
                os.path.join(
                    absolute_path(get_config('main', project_path)['gt_path'], project_path),
                    id2str(cube_id), f'{link["organelle"]}.h5'
                )
            )
            print('      ({})\t{} \t{} \t{}   \t{}'.format(
                link_id, gt_type, link['organelle'], link['image'], 'yes' if is_annotated else 'no'
            ))
    print('')
    print('____________________________________________________________________________________')
    print('')


def log_dataset(dataset, project_path=None):

    trains, vals = get_associated_gt_cubes(dataset, project_path=project_path)

    print('')
    print('____________________________________________________________________________________')
    print(f'DATASET = {dataset}')
    print('')

    print('  TRAIN \tID  \tLayer \tAnnotated')
    print('')
    if trains is None or len(trains) == 0:
        print('      Not linked to any ground truth cubes.')
    else:
        for train_id, train in trains.items():
            print('      {} \t{} \t{} \t{}'.format(
                train_id, train['cube_id'], train['organelle'], 'yes' if train['annotated'] else 'no'
            ))

    print('')
    print('  VAL   \tID  \tLayer \tAnnotated')
    print('')
    if vals is None or len(vals) == 0:
        print('      Not linked to any ground truth cubes.')
    else:
        for val_id, val in vals.items():
            print('      ({})\t{} \t{} \t{}'.format(
                val_id, val['cube_id'], val['organelle'], 'yes' if val['annotated'] else 'no'
            ))

    print('')
    print('____________________________________________________________________________________')
    print('')


def log_datasets(project_path=None):

    print('')
    print('>> DATASETS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    config_main = get_config('main', project_path=project_path)
    # seg_datasets = [x for x in config_main['configs'].keys() if x[:3] == 'seg']
    seg_datasets = [
        x for x in config_main['configs'].keys()
        if (
                x != 'supervoxels'
                and x != 'membrane_prediction'
                and x != 'mask'
                and x != 'raw'
                and x != 'gt'
                and x != 'val'
                and x != 'main'
        )
    ]

    for dataset in seg_datasets:
        log_dataset(dataset, project_path=project_path)

    print('')
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print('')


def log_gt_cubes(val=False, project_path=None):

    print('')
    print('>> CUBES >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    name = 'val' if val else 'gt'
    try:
        gt_config = get_config(name, project_path=project_path)
    except KeyError:
        gt_config = None

    res = get_config('supervoxels', project_path=project_path)['resolution']

    if gt_config is not None:

        for cid, cinfo in gt_config.items():

            log_gt_cube(
                cinfo['id'],
                cinfo['status'],
                (np.array(cinfo['position_center']) * np.array(res))[::-1],
                cinfo['shape'],
                cinfo['links'] if 'links' in cinfo.keys() else None,
                project_path=project_path
            )
    else:
        print('')
        print(f'No {"validation" if val else "ground truth"} cubes found!')

    print('')
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print('')

