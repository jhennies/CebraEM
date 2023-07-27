
import numpy as np
import pickle
import os
import json

from cebra_em_core.project_utils.dependencies import find_dependencies
from cebra_em_core.project_utils.config import get_config, absolute_path
from cebra_em.misc.repo import get_repo_path


def _snakemake_path(project_path):

    snakemake_path = os.path.join(project_path, 'snakemake')
    if not os.path.exists(snakemake_path):
        os.mkdir(snakemake_path)

    return snakemake_path


def generate_run_json(
        targets,
        roi=None,
        unit='px',
        misc=None,
        project_path=None,
        debug=False,
        verbose=False
):

    def _find_target_indices(positions, batch_shape, unit='px', resolution=None, roi=None):

        if roi is not None:
            if unit == 'px':
                start = np.array(roi[:3])
                shp = np.array(roi[3:])
            else:
                assert resolution is not None
                start = np.array(roi[:3]) / resolution
                shp = np.array(roi[3:]) / resolution
            return find_dependencies(start, shp, positions, batch_shape)

        else:
            return list(range(len(positions)))

    def _target_ids_from_roi(target, rois, unit):

        conf_tgt = get_config(target, project_path=project_path)
        batch_shape = conf_tgt['batch_shape']
        positions_fp = absolute_path(conf_tgt['positions'], project_path=project_path)

        with open(positions_fp, 'rb') as f:
            positions = pickle.load(f)

        if unit != 'px':
            res = conf_tgt['resolution']
        else:
            res = None

        ids = []
        for r in rois:
            ids.extend(_find_target_indices(positions, batch_shape, unit=unit, resolution=res, roi=r))

        return ids

    snakemake_path = _snakemake_path(project_path=project_path)
    run_json_fp = os.path.join(snakemake_path, 'run.json')

    if verbose:
        print(f'targets = {targets}')
    if targets == 'gt_cubes' or targets == 'val_cubes':
        run_type = targets
        targets = ['supervoxels']
    elif targets[:7] == 'stitch-':
        run_type = 'stitch'
        targets = [targets[7:]]
    else:
        run_type = 'run'

    with open(run_json_fp, mode='w') as f:
        json.dump(
            {
                'run_type': run_type,
                'targets': targets,
                'target_ids': {tgt: _target_ids_from_roi(tgt, roi, unit) for tgt in targets},
                'verbose': verbose,
                'misc': misc,
                'debug': debug
            }, f, indent=2
        )


def prepare_run_snakefile(targets, project_path, verbose=False):

    if targets == 'gt_cubes' or targets == 'val_cubes':
        targets = ['supervoxels']

    if verbose:
        print(f'targets = {targets}')

    snakefile_path = _snakemake_path(project_path)
    snakefile_fp = os.path.join(snakefile_path, 'run_blocks.smk')

    # Initialize the target snakemake file
    open(snakefile_fp, mode='w').close()

    repo_path = get_repo_path()
    snake_template_path = os.path.join(repo_path, 'snakefiles')

    # Read the template snakefile
    with open(os.path.join(snake_template_path, 'run_block.smk'), mode='r') as f:
        source_block = f.read()

    # -------------------------------------------------------------------------------------
    # Fill the target snakefile with the required code blocks

    ignore_datasets = ['raw']

    def _additional_dependencies(block, dataset, priority):

        conf_ds = get_config(dataset, project_path=project_path)
        additional_inp_str = ''

        if 'add_dependencies' in conf_ds:

            deps = conf_ds['add_dependencies']
            for dep in deps:
                rule_def = dep['rule_def']
                outputs = dep['output']

                if type(outputs) == str:
                    outputs = [outputs]

                for output in outputs:
                    additional_inp_str += f'\n        os.path.join(project_path, "snk_wf", "{output}"),'

                with open(os.path.join(get_repo_path(), 'snakefiles', f'{rule_def}'), mode='r') as f:
                    add_block = f.read()

                add_block.replace('<priority>', str(priority + 9))
                block += add_block

        block = block.replace('<additional_input>', additional_inp_str)
        return block

    def _downstream_dependency(block, dataset, priority):

        conf_ds = get_config(dataset)

        if 'add_downstream' in conf_ds:

            downstreams = conf_ds['add_downstream']
            for downstream in downstreams:
                rule_def = downstream['rule_def']

                with open(os.path.join(get_repo_path(), 'snakefiles', f'{rule_def}'), mode='r') as f:
                    add_block = f.read()

                add_block.replace('<priority>', str(priority))
                block += add_block

        return block

    def _get_resources_str(resources):
        if resources is None:
            return ''
        elif type(resources) == dict:
            rstr = '\n    resources:\n'
            for key, val in resources.items():
                rstr += f'        {key}={val},\n'
            return rstr[:-2]
        else:
            raise ValueError

    def _make_block(target, priority):
        conf_tgt = get_config(target, project_path=project_path)

        run_script = conf_tgt['run_script'] if 'run_script' in conf_tgt else f'run_task.py'
        extension = conf_tgt['extension'] if 'extension' in conf_tgt else 'json'
        resources = _get_resources_str(conf_tgt['resources'])

        block = source_block
        block = _additional_dependencies(block, target, priority)
        block = _downstream_dependency(block, target, priority)
        block = block.replace('<name>', target)
        block = block.replace('<priority>', str(priority))
        block = block.replace('<run_script>', run_script)
        block = block.replace('<resources>', resources)
        block = block.replace('<extension>', extension)

        with open(snakefile_fp, mode='a') as f:
            f.write(block)

    def _make_blocks(targets):

        def _priority_scores(tgts):

            def _merge(*args):
                mge = {}
                for arg in args:
                    for k, v in arg.items():
                        mge[k] = v
                return mge

            scs = []
            for tgt in tgts:
                conf_tgt = get_config(tgt, project_path=project_path)
                deps = conf_tgt['dep_datasets']
                for ignore_ds in ignore_datasets:
                    if ignore_ds in deps:
                        deps.remove(ignore_ds)
                this_scs = _priority_scores(deps)
                sc_tgt = len(deps)
                for dep in deps:
                    sc_tgt += this_scs[dep]
                scs.append(this_scs)
                scs.append({tgt: sc_tgt})
            scs = _merge(*scs)

            return scs

        scores = _priority_scores(targets)

        if verbose:
            print(f'scores = {scores}')

        for image, sc in scores.items():
            _make_block(image, 10 * (sc + 1))

    if verbose:
        print(f'targets = {targets}')
    _make_blocks(targets)

    # -------------------------------------------------------------------------------------


def prepare_run(
        targets,
        roi=None,
        unit='px',
        misc=None,
        project_path=None,
        debug=False,
        verbose=False
):

    if verbose:
        print(f'targets = {targets}')

    # TODO assert that the requested target exists

    # Make sure roi is a list of rois
    if roi is not None:
        if type(roi[0]) != tuple and type(roi[0]) != list:
            roi = [roi]
    else:
        roi = [None]

    # Generates a json file with the basic settings of the current run
    generate_run_json(
        targets,
        roi=roi,
        unit=unit,
        misc=misc,
        project_path=project_path,
        debug=debug,
        verbose=verbose
    )

    # Prepare the snakefiles
    prepare_run_snakefile(targets=targets, project_path=project_path, verbose=verbose)


def find_non_processed_items(items):

    unprocessed = {}
    for key, val in items.items():
        if val['status'] == 'pending':
            unprocessed[key] = val

    return unprocessed


def prepare_gt_extract(project_path=None, verbose=False):

    name = 'gt'

    if verbose:
        print(f'Running {name} extract!')

    def _roi_from_gt_cube_position(item):
        return list(item['position']) + list(item['shape'])

    # Determine gt cubes that are not extracted yet and put them in the queue
    queue = find_non_processed_items(get_config(name, project_path=project_path))

    if verbose:
        print(f'queue = {queue}')

    # Determine the rois for each element in the queue
    roi = [_roi_from_gt_cube_position(val) for key, val in queue.items()]

    if verbose:
        print(f'roi = {roi}')

    prepare_run(f'{name}_cubes', roi=roi, misc=list(queue.keys()), project_path=project_path, verbose=verbose)


def prepare_stitching(
        target,
        beta,
        roi=None,
        unit='px',
        project_path=None,
        verbose=False
):
    if verbose:
        print(f'Running stitching for {target}')

    # Make sure roi is a list of rois
    if roi is not None:
        if type(roi[0]) != tuple and type(roi[0]) != list:
            roi = [roi]
    else:
        roi = [None]

    generate_run_json(
        target,
        roi=roi,
        unit=unit,
        misc=dict(beta=beta),
        project_path=project_path,
        verbose=verbose
    )

    snakefile_path = _snakemake_path(project_path)
    snakefile_fp = os.path.join(snakefile_path, 'stitch_segmentation.smk')
    repo_path = get_repo_path()
    snake_template_path = os.path.join(repo_path, 'snakefiles')

    src = os.path.join(snake_template_path, 'stitch_segmentation.smk')
    tgt = snakefile_fp

    # Read the template snakefile
    with open(src, mode='r') as f:
        source_block = f.read()

    source_block = source_block.replace('<name>', target[7:])

    print(source_block)

    # Write the result snakefile
    with open(tgt, mode='w') as f:
        f.write(source_block)
