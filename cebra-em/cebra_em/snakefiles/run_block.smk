
# _________________________________________________________________________________________
# <name>
import os

config_<name> = get_config('<name>', project_path=project_path)
resources_<name> = config_<name>['resources']


def _make_dependencies_<name>(wildcards):
    dep_datasets = get_config('<name>')['dep_datasets']
    with open(os.path.join(project_path, 'tasks', 'dependencies_<name>.pkl'), 'rb') as f:
        dependencies = pickle.load(f)
    dependency_list = []
    # Only using the last dependency here, otherwise snakemake exhibited some unforseen behavior
    # which I still don't quite understand...
    dep_ds_idx = len(dep_datasets) - 1
    dep_ds = dep_datasets[-1]
    # # This is the original loop, which technically should be correct but doesn't work for some reason:
    # for dep_ds_idx, dep_ds in enumerate(dep_datasets):
    #     if dependencies[dep_ds_idx] is not None: ...
    if dependencies[dep_ds_idx] is not None:
        assert dep_ds != 'raw'
        for dep_idx in dependencies[dep_ds_idx][int(wildcards.idx)]:
            dependency_list.append(ancient(os.path.join(project_path, "snk_wf", f"run_{dep_ds}_{dep_idx}.json")))
    if 'quantile_norm' in config_<name> and config_<name>['quantile_norm'] is not None:
        dependency_list.append(os.path.join(project_path, "snk_wf", f"raw_quantiles.json"))
    return dependency_list


rule run_<name>:
    priority: <priority> + 2
    input: <additional_input>
        _make_dependencies_<name>
    output:
        os.path.join(project_path, "snk_wf", "run_<name>_{idx}.<extension>")
    resources:
        gpu=resources_<name>['gpus'],
        cpus=resources_<name>['cpus'],
        mem_mb=resources_<name>['mem_mb'],
        time_min=resources_<name>['time_min']
    threads: resources_<name>['cpus']
    params:
        image_name="<name>",
        p='gpu' if resources_<name>['gpus'] > 0 else 'htc',
        gres='--gres=gpu:1' if resources_<name>['gpus'] > 0 else ''
        # gres='--gres=gpu:1 -C gpu=A40' if resources_<name>['gpus'] > 0 else ''
    script:
        os.path.join(cebra_em_path, 'run_scripts', '<run_script>')
