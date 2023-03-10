
from cebra_em.run_utils.run_specs import get_run_json
from cebra_em_core.project_utils.project import get_current_project_path
from cebra_em_core.project_utils.gt import extract_gt

import re


if __name__ == '__main__':

    print(f">>> STARTING: gt_extract for {snakemake.wildcards['cube_id']}")

    # _______________________________________________________________________________
    # Retrieving settings

    project_path = get_current_project_path(None)
    run_json = get_run_json(project_path)

    verbose = run_json['verbose']
    run_type = run_json['run_type']

    cube_id = snakemake.wildcards['cube_id']
    cube_id_int = int(re.findall('\d+', cube_id)[0])

    outputs = snakemake.output

    # _______________________________________________________________________________
    # Run the task (results are written internally)

    extract_gt(
        cube_id,
        raw_fp=outputs[0],
        mem_fp=outputs[1],
        sv_fp=outputs[2],
        project_path=project_path,
        verbose=verbose
    )

    # _______________________________________________________________________________
    # Update the gt config file



