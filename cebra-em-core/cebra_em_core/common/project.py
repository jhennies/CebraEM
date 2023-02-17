
import os
import json
from .version import __version__, VALID_VERSIONS


def assert_valid_project(project_path):

    proj_json = os.path.join(project_path, 'project.json')
    if not os.path.isfile(proj_json):
        print('Not a valid project location: No project.json found')
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
