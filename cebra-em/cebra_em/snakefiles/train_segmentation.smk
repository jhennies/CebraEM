
n_workers = config_<name>['learning_kwargs']['n_jobs']
train_cubes, val_cubes = get_associated_gt_cubes('<name>', project_path=project_path)
if verbose:
    print(f'train_cubes = {train_cubes}')

train_cube_files = [
    f'{key}/{val["organelle"]}.h5'
    for key, val in train_cubes.items()
]

rule train_<name>:
    priority: <priority> + 9
    input:
        expand(os.path.join(project_path, "gt", "{cube_file}"), cube_file=train_cube_files),
        expand(os.path.join(project_path, "gt", "{cube_id}", "{inp}.h5"), cube_id=train_cubes.keys(), inp=['raw', 'mem', 'sv'])
    output:
        os.path.join(project_path, "snk_wf", "train_<name>_rf.pkl"),
        os.path.join(project_path, "snk_wf", "train_<name>_nrf.pkl")
    threads: n_workers
    resources:
        cpus=n_workers, mem_mb=16000, time_min=60, gpus=0
    params:
        image_name='<name>',
        p='htc',
        gres=''
    script:
        os.path.join(cebra_em_path, 'run_scripts','train_segmentation.py')

