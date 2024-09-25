
import os
import json
from cebra_em_core.version import __version__, VALID_VERSIONS


def get_project_json(project_path):
    return os.path.join(project_path, 'cebra-em.json')


def assert_valid_project(project_path):

    proj_json = get_project_json(project_path)
    if not os.path.isfile(proj_json):
        print(f'Not a valid project location: {project_path}')
        raise RuntimeError('Not a valid project location: No project.json found')
    with open(proj_json) as f:
        proj_info = json.load(f)
    if proj_info['type'] != 'cebra_em':
        print(f'Invalid project type: {proj_info["type"]}')
        raise RuntimeError(f'Invalid project type: {proj_info["type"]}')
    if proj_info['em_core_version'] not in VALID_VERSIONS:
        print(f'Invalid em_core_version: {proj_info["em_core_version"]}')
        print(f'Valid versions: {VALID_VERSIONS}')
        raise RuntimeError(f'Invalid em_core_version: {proj_info["em_core_version"]}')
    return True


def make_project_structure(project_path, ignore_non_empty=False):
    if not os.path.exists(project_path):
        os.mkdir(project_path)
    else:
        if not ignore_non_empty:
            assert not os.listdir(project_path), 'Project path exists and is not empty!'

    # Make project json
    with open(get_project_json(project_path), mode='w') as f:
        json.dump({
            'em_core_version': __version__,
            'type': 'cebra_em'
        }, f)

    config_path = os.path.join(project_path, 'config')
    tasks_path = os.path.join(project_path, 'tasks')
    params_path = os.path.join(project_path, 'params')
    workflow_path = os.path.join(project_path, 'snk_wf')
    if not os.path.exists(config_path):
        os.mkdir(config_path)
    if not os.path.exists(tasks_path):
        os.mkdir(tasks_path)
    if not os.path.exists(params_path):
        os.mkdir(params_path)
    if not os.path.exists(workflow_path):
        os.mkdir(workflow_path)

    # Add a mock input file used by run_block.smk
    open(os.path.join(workflow_path, 'mock'), mode='w').close()

    return config_path, params_path, tasks_path


def get_current_project_path(project_path=None):

    if project_path is None:
        project_path = os.path.join(os.path.abspath('.'), '')
    else:
        project_path = os.path.join(os.path.abspath(project_path), '')
    assert_valid_project(project_path)

    return project_path


def lock_project(project_path=None):
    project_path = get_current_project_path(project_path=project_path)

    lock_fp = os.path.abspath(os.path.join(project_path, '.lock'))
    if os.path.exists(lock_fp):
        return lock_fp, 'is_locked_error'
    else:
        try:
            open(lock_fp, 'w').close()
        except:
            return lock_fp, 'could_not_lock_error'
        return lock_fp, ''


def unlock_project(project_path=None):
    project_path = get_current_project_path(project_path=project_path)

    lock_fp = os.path.join(project_path, '.lock')

    if os.path.exists(lock_fp):
        try:
            os.remove(lock_fp)
        except:
            return 'could_not_unlock_error'
        return ''
    else:
        return 'is_unlocked_error'


def delete_segmentation_config(name, project_path=None, verbose=False, debug=False):
    from cebra_em_core.project_utils.config import get_config_filepath

    # Removing segmentation config
    config_filepath = get_config_filepath(name, project_path)

    if verbose:
        print(f'config_filepath = {config_filepath}')

    import os

    try:
        print(f'Deleting: {config_filepath}')
        if not debug:
            os.remove(config_filepath)
    except Exception as e:
        print(f'Error deleting {config_filepath}: {e}')


def remove_config_link(name, project_path=None, verbose=False, debug=False):

    from cebra_em_core.project_utils.config import get_config, get_config_filepath
    config_main = get_config('main', project_path)

    if verbose:
        print('')
        for k, v in config_main['configs'].items():
            print(f'{k}: {v}')
        print('')

    del(config_main['configs'][name])

    if verbose:
        for k, v in config_main['configs'].items():
            print(f'{k}: {v}')
        print('')

    if not debug:
        config_main_fp = get_config_filepath('main', project_path)
        with open(config_main_fp, 'w') as f:
            json.dump(config_main, f, indent=2)


def remove_segmentation_meta(name, project_path=None, verbose=False, debug=False):

    print(f'Cleaning up metadata for: {name}\n')

    delete_segmentation_config(name, project_path=project_path, verbose=verbose, debug=debug)
    remove_config_link(name, project_path=None, verbose=verbose, debug=debug)







