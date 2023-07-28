
import os
import json
import numpy as np
from cebra_em_core.project_utils.project import get_current_project_path
from cebra_em_core.project_utils.config import get_config
from cebra_em_core.project_utils.gt import get_associated_gt_cubes
from cebra_em.misc.repo import get_repo_path
from cebra_em.run_utils.run_specs import get_run_json
import pickle  # Required in run_block.smk for loading of dependencies

cebra_em_path = get_repo_path()
project_path = get_current_project_path(None)
print(f'project_path = {os.path.abspath(project_path)}')

run_info = get_run_json(project_path)

run_type = run_info['run_type']
targets = run_info['targets']
indices = run_info['target_ids']
verbose = run_info['verbose']

if run_type == 'run':

    print(f'len(targets) = {len(targets)}')

    target_outputs = []
    for target in targets:
        config_target = get_config(target)
        ext = config_target['extension'] if 'extension' in config_target else 'json'
        target_outputs.append(
            os.path.join(
                project_path,
                'snk_wf',
                '{out}'.format(out='run_{dataset}_{idx}.' + f'{ext}')
            )
        )
    print(f'target_outputs = {target_outputs}')
    print(f'targets = {targets}')

    def _expand_inputs(wildcards):
        inp_list = []
        for idx in indices[targets[0]]:
            # print(f'idx = {idx}')
            for target in targets:
                config_target = get_config(target)
                # print(f'target = {target}')
                if 'add_downstream' in config_target:
                    if 'param' in config_target['add_downstream'][0]:
                        param_id = str.split(config_target['add_downstream'][0]['param'], ':')
                        param_val = config_target
                        for pid in param_id:
                            param_val = param_val[pid]
                    if type(param_val) == list:
                        for pv in param_val:
                            inp_list.append(
                                os.path.join(
                                    project_path,
                                    'snk_wf',
                                    config_target['add_downstream'][0]['output'].format(param=pv, dataset=target, idx=idx)
                                )
                            )
                    else:
                        inp_list.append(
                            os.path.join(
                                project_path,
                                'snk_wf',
                                config_target['add_downstream'][0]['output'].format(param=param_val,dataset=target,idx=idx)
                            )
                        )
                else:
                    inp_list.append(
                        os.path.join(
                            project_path,
                            'snk_wf',
                            f'run_{target}_{idx}.' + f'{ext}'
                        )
                    )
        for x in sorted(inp_list):
            print(x)
        return inp_list

    rule all:
        input:
            _expand_inputs
        params: p='htc', gres=''

    include: os.path.join(project_path, 'snakemake', 'run_blocks.smk')

elif run_type == 'gt_cubes':

    name = 'gt'

    target_output = os.path.join(
            project_path,
            'snk_wf',
            '{out}'.format(out='run_{dataset}_{idx}.json')
    )

    assert len(targets) == 1, 'targets for run_type = "gt_cubes" should be only "supervoxels"' \
                              f'instead it is {targets}'
    assert targets[0] == 'supervoxels', 'targets for run_type = "gt_cubes" should be only "supervoxels"' \
                                        f'instead it is {targets}'

    # The ground truth cube ids as stored in the miscellaneous field of the run json
    gt_cube_ids = run_info['misc']

    rule all:
        input:
            expand(os.path.join(project_path, name, "{cube_id}", "raw.h5"), cube_id=gt_cube_ids)

    rule extract_gt:
        input:
            expand(target_output, dataset=targets[0], idx=indices[targets[0]])
        output:
            os.path.join(project_path, name, "{cube_id}", "raw.h5"),
            os.path.join(project_path, name, "{cube_id}", "mem.h5"),
            os.path.join(project_path, name, "{cube_id}", "sv.h5")
        resources:
            cpus=1, time_min=10, mem_mb=4096
        params: p='htc', gres=''
        script:
            os.path.join(cebra_em_path, 'run_scripts','../run_scripts/run_extract_gt.py')

    include: os.path.join(project_path, 'snakemake', 'run_blocks.smk')

elif run_type == 'stitch':
    print(run_info)

    assert len(targets) == 1
    target = targets[0]

    rule all:
        input:
            expand(
                os.path.join(
                    project_path,
                    'snk_wf',
                    'apply_mapping_{target}_{idx}.done'.format(target=target, idx='{idx}')
                ),
                idx=indices[targets[0]]
            )

    include: os.path.join(project_path, 'snakemake', 'stitch_segmentation.smk')

else:
    raise ValueError('Invalid run type')


rule quantile_norm:
    output:
        os.path.join(os.path.join(project_path, "snk_wf", f"raw_quantiles.json"))
    resources:
        cpus=os.cpu_count(), time_min=60, mem_mb=32000
    params: p='htc', gres=''
    script:
        os.path.join(cebra_em_path, 'run_scripts', 'run_quantile_norm.py')
