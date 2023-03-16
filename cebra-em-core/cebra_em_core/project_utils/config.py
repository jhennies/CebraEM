
import os
import json
import numpy as np
from cebra_em_core.project_utils.project import get_current_project_path
from cebra_em_core.project_utils.params import load_params


def absolute_path(path, project_path=None):
    project_path = get_current_project_path(project_path=project_path)
    return path.format(project_path=project_path)


def get_mask_xml(project_path=None):
    try:
        return get_config('mask', project_path=project_path)['xml_path']
    except KeyError:
        return None


def get_config_path(relpath=False, project_path=None):
    project_path = get_current_project_path(project_path=project_path)

    if relpath:
        return 'config'
    else:
        return os.path.join(project_path, 'config')


def get_config_filepath(image_name, project_path=None):

    with open(os.path.join(get_config_path(project_path=project_path), 'config_main.json'), 'r') as f:
        config_main = json.load(f)
    return absolute_path(config_main['configs'][image_name], project_path=project_path)


def get_config(image_name, project_path=None):
    config_fp = get_config_filepath(image_name, project_path=project_path)
    with open(config_fp, 'r') as f:
        conf = json.load(f)
    return conf


def extend_dict(base_dict, in_dict):

    for k, v in in_dict.items():
        if type(v) == dict:
            if k not in base_dict.keys():
                base_dict[k] = {}
            extend_dict(base_dict[k], v)
        elif type(v) == np.ndarray:
            base_dict[k] = v.tolist()
        else:
            base_dict[k] = v

    return base_dict


def add_to_config_json(filename, data, verbose=False):

    if verbose:
        print(f'Adding to {filename}:')
        print(data)

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    config = extend_dict(config, data)

    # for k, v in data.items():
    #     if k not in config.keys():
    #         config[k] = {}
    #     if type(v) == dict:
    #         for kk, vv in v.items():
    #             if type(vv) == np.ndarray:
    #                 config[k][kk] = vv.tolist()
    #             else:
    #                 config[k][kk] = vv
    #     else:
    #         if type(v) == np.ndarray:
    #             config[k] = v.tolist()
    #         else:
    #             config[k] = v

    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)


def init_image_config(image_name, project_path=None, force=False):

    if not force:
        assert image_name not in get_config('main', project_path=project_path)['configs']
    config_main_fp = get_config_filepath('main', project_path=project_path)

    config_image_fp = os.path.join(get_config_path(project_path=project_path), f'config_{image_name}.json')
    config_image_rel = os.path.join(get_config_path(relpath=True, project_path=project_path), f'config_{image_name}.json')

    add_to_config_json(config_main_fp, {'configs': {image_name: '{project_path}' + config_image_rel}})
    add_to_config_json(
        config_image_fp,
        load_params('general', project_path=project_path)
    )
    add_to_config_json(
        config_image_fp,
        load_params(image_name, project_path=project_path)
    )


def init_mask_config(mask_xml, project_path=None, force=False):

    if not force:
        assert 'mask' not in get_config('main', project_path=project_path)['configs']
    config_main_fp = get_config_filepath('main', project_path=project_path)

    config_mask_fp = os.path.join(get_config_path(project_path=project_path), 'config_mask.json')
    config_mask_rel = os.path.join(get_config_path(relpath=True, project_path=project_path), 'config_mask.json')

    add_to_config_json(config_main_fp, {'configs': {'mask': '{project_path}' + config_mask_rel}})
    add_to_config_json(config_mask_fp, {'xml_path': mask_xml})
    add_to_config_json(config_mask_fp, load_params('mask', project_path=project_path))


def init_raw_config(raw_data_xml, project_path=None, force=False):

    if not force:
        assert 'raw' not in get_config('main', project_path=project_path)['configs']
    config_main_fp = get_config_filepath('main', project_path=project_path)

    config_raw_fp = os.path.join(get_config_path(project_path=project_path), 'config_raw.json')
    config_raw_rel = os.path.join(get_config_path(relpath=True, project_path=project_path), 'config_raw.json')

    add_to_config_json(config_main_fp, {'configs': {'raw': '{project_path}' + config_raw_rel}})
    add_to_config_json(config_raw_fp, {'xml_path': raw_data_xml})
    add_to_config_json(config_raw_fp, load_params('raw', project_path=project_path))


def init_main_config(project_path=None, verbose=False):

    config_fp = os.path.join(get_config_path(project_path=project_path), 'config_main.json')
    config_rel = os.path.join(get_config_path(relpath=True, project_path=project_path), 'config_main.json')
    add_to_config_json(
        config_fp,
        dict(
            project_path=project_path,
            tasks_path='{project_path}tasks',
            params_path='{project_path}params',
            mobie_project_path='{project_path}data',
            configs=dict(
                main='{project_path}' + config_rel
            )
        ),
        verbose=verbose
    )


def set_version(version, project_path=None):
    config_main_fp = get_config_filepath('main', project_path=project_path)
    add_to_config_json(
        config_main_fp,
        {'version': version}
    )
