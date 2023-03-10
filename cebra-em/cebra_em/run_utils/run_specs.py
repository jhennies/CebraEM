
import os
import json
from cebra_em_core.project_utils.project import get_current_project_path


def get_run_json_fp(project_path=None):
    project_path = get_current_project_path(project_path=project_path)

    return os.path.join(
        project_path,
        'snakemake',
        'run.json'
    )


def get_run_json(project_path=None):

    fp = get_run_json_fp(project_path)

    with open(fp, mode='r') as f:
        run_json = json.load(f)

    return run_json
