
import sys
import numpy as np
import pickle
import os
from concurrent.futures import ThreadPoolExecutor

from cebra_em_core.project_utils.config import get_config, get_config_filepath, absolute_path, add_to_config_json


def find_dependencies(
        position, shape, dep_positions, dep_shape
):
    """
    Finds the ids of overlapping volumes with a specified volume
    :param position: The position of the volume
    :param shape: The shape of the volume
    :param dep_positions: The positions of the potentially overlapping volumes
    :param dep_shape: The shape of the potentially overlapping volumes
    :return: list of ids of overlapping volumes
    """

    # Do the full search for the first dimension
    x = np.logical_and(position[0] - dep_shape[0] < dep_positions[:, 0],
                       dep_positions[:, 0] < position[0] + shape[0])
    x = np.where(x)

    # Only search within the candidates of the previous dimension
    y_pos = dep_positions[:, 1][x]
    y = np.logical_and(position[1] - dep_shape[1] < y_pos, y_pos < position[1] + shape[1])
    y = x[0][np.where(y)]

    # And again narrowing it down for the last dimension
    z_pos = dep_positions[:, 2][y]
    z = np.logical_and(position[2] - dep_shape[2] < z_pos, z_pos < position[2] + shape[2])
    z = y[np.where(z)]

    return z.tolist()


def depends_on_nothing(pos):
    return None


def depends_on_raw(pos):
    return None


def depends_on_something(
        name,
        shape,
        dep_positions,
        dep_shape,
        halo
):

    def dos(pos):

        return find_dependencies(pos - halo, shape + 2 * halo, dep_positions, dep_shape)

    return dos


def make_dependencies(
        positions,
        depends_on=depends_on_nothing,
        n_workers=1,
        verbose=False
):

    if depends_on == depends_on_nothing or depends_on == depends_on_raw:

        dependencies = None

    else:

        def _print_status(idx, pos):
            sys.stdout.write('\r' + 'Generating tasks {}/{}'.format(idx + 1, len(positions)))
            return depends_on(pos)

        if n_workers == 1:
            dependencies = [
                _print_status(idx, position)
                for idx, position in enumerate(positions)
            ]
        else:

            with ThreadPoolExecutor(max_workers=n_workers) as tpe:

                tasks = [
                    tpe.submit(_print_status, idx, position)
                    for idx, position in enumerate(positions)
                ]

                dependencies = [
                    task.result() for task in tasks
                ]

        print('')
        print('')

    return dependencies


def init_dependencies(image_name, project_path=None, n_workers=1, verbose=False):

    # _______________________________________________________________________________
    # Retrieving settings

    config_fp = get_config_filepath(image_name, project_path=project_path)
    config = get_config(image_name, project_path=project_path)
    positions_fp = absolute_path(config['positions'], project_path=project_path)
    dep_ds = config['dep_datasets']
    batch_shape = config['batch_shape']
    ds_shape = config['shape']
    halo = config['halo']

    if dep_ds is not None:
        dep_batch_shapes = []
        dep_positions_fps = []
        for dd in dep_ds:
            if dd != 'raw':
                config_dep = get_config(dd, project_path=project_path)
                dep_batch_shapes.append(config_dep['batch_shape'])
                dep_positions_fps.append(absolute_path(config_dep['positions'], project_path=project_path))
            else:
                dep_batch_shapes.append(None)
                dep_positions_fps.append(None)

    if verbose:
        print(f'image_name = {image_name}')
        print(f'project_path = {project_path}')
        print(f'positions_fp = {positions_fp}')
        print(f'dep_ds = {dep_ds}')
        print(f'batch_shape = {batch_shape}')
        print(f'ds_shape = {ds_shape}')
        print(f'halo = {halo}')

    # _______________________________________________________________________________
    # Fill the task dictionary
    with open(positions_fp, 'rb') as f:
        positions = pickle.load(f)
    if dep_ds is not None:
        dep_positions = []
        for didx, dd in enumerate(dep_ds):
            if dd != 'raw':
                with open(dep_positions_fps[didx], 'rb') as f:
                    dep_positions.append(pickle.load(f))
            else:
                dep_positions.append(None)
    if verbose:
        if dep_ds is not None:
            print(f'positions = {positions}')
            for didx, dd in enumerate(dep_ds):
                print(f'dep_ds = {dd}, dep_positions = {dep_positions[didx]}')

    if dep_ds is not None:
        dependencies = []
        for ds_idx, ds in enumerate(dep_ds):
            if ds is None:
                depends_on_x = depends_on_nothing
            elif ds == 'raw':
                depends_on_x = depends_on_raw
            else:
                depends_on_x = depends_on_something(
                    dep_ds,
                    np.array(batch_shape),
                    dep_positions[ds_idx],
                    dep_batch_shapes[ds_idx],
                    np.array(halo)
                )
            dependencies.append(
                make_dependencies(
                    positions,
                    depends_on=depends_on_x,
                    n_workers=n_workers,
                    verbose=verbose
                )
            )
    else:
        dependencies = None

    dependencies_fp = os.path.join('{project_path}tasks', f'dependencies_{image_name}.pkl')
    with open(absolute_path(dependencies_fp, project_path=project_path), 'wb') as f:
        pickle.dump(dependencies, f)

    add_to_config_json(
        config_fp,
        {
            'dependencies': dependencies_fp
        }
    )

