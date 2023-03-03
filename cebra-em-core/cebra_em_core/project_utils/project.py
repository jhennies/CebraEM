
import os
import json
from cebra_em_core.version import __version__, VALID_VERSIONS


def assert_valid_project(project_path):

    proj_json = os.path.join(project_path, 'project.json')
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
    with open(os.path.join(project_path, 'project.json'), mode='w') as f:
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
