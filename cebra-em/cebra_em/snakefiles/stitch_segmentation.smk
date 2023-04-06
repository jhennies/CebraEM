
rule find_mapping_<name>:
    priority: 200
    output:
        os.path.join(project_path, "snk_wf", "find_mapping_<name>_{idx}.json")
    resources:
        cpus=1, mem_mb=1024, time_min=10, gpus=0
    params:
        image_name="<name>",
        p='htc',
        gres=''
    script:
        os.path.join(cebra_em_path, 'run_scripts', 'find_mapping.py')

rule solve_mapping_<name>:
    priority: 200
    input:
        expand(os.path.join(project_path, "snk_wf", "find_mapping_{dataset}_{idx}.json"), dataset='<name>', idx=indices['<name>'])
    output:
        os.path.join(project_path, "snk_wf", "solve_mapping_<name>.json")
    resources:
        cpus=1, mem_mb=1024, time_min=30, gpus=0
    params:
        image_name="<name>",
        p='htc',
        gres=''
    script:
        os.path.join(cebra_em_path, 'run_scripts', 'solve_mapping.py')

rule apply_mapping_<name>:
    priority: 200
    input:
        os.path.join(project_path, "snk_wf", "solve_mapping_<name>.json")
        # "snm_wf/setup_stitched_dataset_<name>.done"
    output:
        os.path.join(project_path, "snk_wf", "apply_mapping_<name>_{idx}.done")
    resources:
        cpus=1, mem_mb=4096, time_min=30, gpu=0
    params:
        image_name="<name>",
        p='htc',
        gres=''
    script:
        os.path.join(cebra_em_path, 'run_scripts', 'apply_mapping.py')
