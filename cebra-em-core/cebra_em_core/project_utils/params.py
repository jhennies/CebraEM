
import sys
import os
from shutil import copy
import json
from cebra_em_core.misc.repo import get_repo_path
from cebra_em_core.project_utils.project import get_current_project_path


if sys.platform == 'linux' or sys.platform == 'darwin':
    from cebra_em_core.linux.misc import show_in_editor
elif sys.platform[:3] == 'win':
    from cebra_em_core.windows.misc import show_in_editor
else:
    # raise RuntimeError(f'Running on {sys.platform} not supported! ')
    print(f'Warning: Running on {sys.platform} might not be supported!')
    from cebra_em_core.linux.misc import show_in_editor


def get_params_path(image_name, project_path=None):
    project_path = get_current_project_path(project_path=project_path)
    return os.path.join(project_path, 'params', f'{image_name}.json')


def copy_default_params(
        params=('mask', 'membrane_prediction', 'supervoxels'),
        target_names=None,
        project_path=None,
        verbose=False
):

    if target_names is None:
        target_names = params

    defaults_path = os.path.join(
        get_repo_path(),
        'project_utils', 'params'
    )
    if verbose:
        print(f'defaults_path = {defaults_path}')

    for idx, param in enumerate(params):

        dst = get_params_path(target_names[idx], project_path=project_path)
        if verbose:
            print(f'dst = {dst}')
        if not os.path.exists(dst):
            src = os.path.join(defaults_path, f'{param}_defaults.json')
            copy(src, dst)


def load_params(image_name, project_path=None):

    with open(get_params_path(image_name, project_path=project_path), mode='r') as f:
        return json.load(f)


def query_parameters(all_params, project_path=None):
    # Find out which parameters are to be queried
    params_to_query = []
    for key, params in all_params.items():
        if params is None:
            params_to_query.append(key)
            # all_params[key] = load_params(key, project_path=project_path)

    def _display_dict(dct, indent=0):
        dct_str = '{\n'
        for key, val in dct.items():
            if type(val) == dict:
                dct_str += f'{"    " + "    " * indent}{key}: {_display_dict(val, indent + 1)}'
            else:
                dct_str += f'{"    " + "    " * indent}{key}: {val}\n'
        dct_str += '    ' * indent + '}\n'
        return dct_str

    def _make_query_string_item(name, params):
        query_str = (
            f'{name}:\n'
            f'{_display_dict(params)}\n'
        )
        return query_str

    def _make_question_string(params):
        param_strings = [f'[{pidx}] {param}' for pidx, param in enumerate(params)]
        return f'Change parameters: {", ".join(param_strings)}, [ENTER] to continue  '

    def _query_character(question, letters, enter=False):

        if enter:
            valid = letters + ['']
        else:
            valid = letters

        while True:
            sys.stdout.write(question)
            choice = input().lower()
            if choice in valid:
                return choice
            else:
                if enter:
                    sys.stdout.write(f'\nPlease respond with {", ".join(letters)} or press ENTER\n\n')
                else:
                    sys.stdout.write(f'\nPlease respond with {", ".join(letters)}\n\n')

    if len(params_to_query) > 0:

        answer = '0'

        while answer != '':

            # Generate the display string
            query_string = '\nPARAMETERS\n\n'
            for key in params_to_query:
                all_params[key] = load_params(key, project_path=project_path)
                query_string += _make_query_string_item(key, all_params[key])

            # Show all parameters
            print(query_string)

            # Ask the question and wait for user input
            answer = _query_character(
                _make_question_string(params_to_query),
                [str(x) for x in range(len(params_to_query))],
                enter=True
            )

            # If user didn't just press enter, open the respective parameters in the editor
            if answer != '':
                show_in_editor(get_params_path(params_to_query[int(answer)], project_path=project_path))

    return all_params

