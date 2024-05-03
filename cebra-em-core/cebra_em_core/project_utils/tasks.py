
import os
import pickle
import numpy as np
from pybdv.util import get_key, get_scale_factors, open_file
from pybdv.metadata import get_data_path
from cebra_em_core.project_utils.config import get_config, get_config_filepath, absolute_path, add_to_config_json
from cebra_em_core.dataset.bdv_utils import is_h5


def get_tasks_folder(project_path=None, relpath=False):

    tasks_rel_path = get_config('main', project_path=project_path)['tasks_path']

    if relpath:
        return tasks_rel_path
    else:
        return tasks_rel_path.format(project_path=project_path)


def pos_generator(
        shp, full_shp, res, offset,
        mask_xml_path=None,
        mask_res=None,
        mask_ds_level=0,
        mode=('min_value_count', {'min_value': 10}),
        dilate_mask=False,
        verbose=False
):
    import sys

    def _get_mask(xml_path, ds_lvl, res):

        data_path = get_data_path(xml_path, return_absolute_path=True)
        key = get_key(is_h5(xml_path), 0, 0, ds_lvl)

        with open_file(data_path, mode='r') as f:
            m = f[key][:]

        # Determine the resolution (mask_res is the resolution of ds_level=0)
        scale_factor = get_scale_factors(data_path, 0)[ds_lvl]
        print(f'Before correction: res = {res}')
        res = np.array(res) * np.array(scale_factor)
        print(f'After: scale_factor = {scale_factor} | res = {res}')
        return m, res

    def _binarize(m, method, args):

        if method == 'binary':
            # The conversion to bool below already takes care of it
            pass
        elif method == 'upper_lower':
            # Good when the raw data is used as mask as well
            m[m > args['upper']] = 0
            m[m < args['lower']] = 0
        elif method == 'label_map':
            # For when a cell segmentation is available and specific cells should be used
            tmp_m = np.zeros(m.shape, dtype=bool)
            for lbl in args['ids']:
                tmp_m = tmp_m | (m == lbl)
            m = tmp_m
        else:
            raise ValueError(f'Method not recognized: {method}')

        m = m.astype(bool)

        return m

    def _remove_outside_mask(positions, mxml, mdsl, mres, dres, dshp):

        m, mres = _get_mask(mxml, mdsl, mres)
        m = _binarize(m, mode[0], mode[1]).astype('uint8') * 255

        p_mask = np.zeros((positions.shape[0],), dtype=bool)
        td, th, tw = np.ceil(np.array(dshp) * np.array(dres) / np.array(mres)).astype(int)

        print('')
        if verbose:
            print(f'len(positions) = {len(positions)}')
        for pidx, p in enumerate(positions):

            sys.stdout.write('\r' + f'{100 * pidx / (len(positions) - 1)} %: position = {p}')

            tz, ty, tx = (np.array(p) * np.array(dres) / np.array(mres)).astype(int)

            ttd, tth, ttw = td, th, tw
            if tz < 0 or ty < 0 or tx < 0:
                if tz < 0:
                    ttd = td + tz
                    tz = 0
                if ty < 0:
                    tth = th + ty
                    ty = 0
                if tx < 0:
                    ttw = tw + tx
                    tx = 0

            if verbose:
                print(f'tpos = {[tz, ty, tx]}')
                print(f'mshp = {[td, th, tw]}')
                print(f'mshp_ = {[ttd, tth, ttw]}')

            val = m[tz: tz + ttd, ty: ty + tth, tx: tx + ttw].max()

            if verbose:
                print(f'val = {val}')

            p_mask[pidx] = val > 0

        print('')

        if verbose:
            print(f'p_mask = {p_mask}')

        return positions[p_mask]

    if verbose:
        print('Making positions:')
        print(f'shp = {shp}')
        print(f'full_shp = {full_shp}')
        print(f'res = {res}')
        print(f'offset = {offset}')
        print(f'mask_xml_path = {mask_xml_path}')
        print(f'mask_ds_level = {mask_ds_level}')
        print(f'mode = {mode}')

    full_shp = np.array(full_shp)
    offset = np.array(offset)
    shp = np.array(shp)

    # Make pos meshgrid
    mesh_shp = np.ceil((full_shp + offset) / shp).astype(int)
    if verbose:
        print(f'mesh_shp = {mesh_shp}')
    pos = [[], [], []]
    pos[1], pos[0], pos[2] = np.meshgrid(
        range(mesh_shp[1]),
        range(mesh_shp[0]),
        range(mesh_shp[2])
    )

    # Flatten the position array
    pos = np.array(pos)
    pos_target_shape = np.prod(pos.shape[1:])
    pos = pos.reshape((3, pos_target_shape)).swapaxes(0, 1)
    if verbose:
        print(f'pos.shape = {pos.shape}')

    # Positions to original resolution
    pos = pos * shp[None, :] - offset

    # Clean up positions that do not intersect with the mask
    if mask_xml_path is not None:
        pos = _remove_outside_mask(pos, mask_xml_path, mask_ds_level, mask_res, res, shp)

    return pos


def compute_task_positions(image_name, project_path=None, verbose=False):

    # _______________________________________________________________________________
    # Retrieving settings

    tasks_rel_folder = get_tasks_folder(project_path=project_path, relpath=True)
    tasks_folder = absolute_path(tasks_rel_folder, project_path=project_path)
    config = get_config(image_name, project_path=project_path)
    config_fp = get_config_filepath(image_name, project_path=project_path)

    if verbose:
        print(f'image_name = {image_name}')
        print(f'config = {config}')

    batch_shape = config['batch_shape']
    ds_shape = config['shape']
    target_resolution = config['resolution']
    offset = config['offset']

    try:
        config_mask = get_config('mask', project_path=project_path)

        mask_xml_path = config_mask['xml_path']
        mask_resolution = config_mask['resolution']
        mask_ds_level_for_init = config_mask['ds_level_for_init']
        mask_method = config_mask['method']
        mask_args = config_mask['args']
        dilate_mask = config_mask['dilate']
    except KeyError:
        if verbose:
            print('No mask found!')
        mask_xml_path = None
        mask_resolution = None
        mask_ds_level_for_init = None
        mask_method = None
        mask_args = None
        dilate_mask = False

    # _______________________________________________________________________________
    # Compute positions

    assert os.path.exists(tasks_folder), f'Tasks folder does not exist: {tasks_folder}'

    positions = pos_generator(
        batch_shape,
        ds_shape,
        target_resolution,
        offset,
        mask_xml_path=mask_xml_path,
        mask_res=mask_resolution,
        mask_ds_level=mask_ds_level_for_init,
        mode=(mask_method, mask_args),
        dilate_mask=dilate_mask,
        verbose=verbose
    )

    positions_rel_fp = os.path.join(tasks_rel_folder, f'positions_{image_name}.pkl')
    positions_fp = absolute_path(positions_rel_fp, project_path=project_path)
    with open(positions_fp, 'wb') as f:
        pickle.dump(positions, f)

    add_to_config_json(
        config_fp,
        {
            'positions': positions_rel_fp
        }
    )

