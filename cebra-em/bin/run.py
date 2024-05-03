
def _snakemake(*args, **kwargs):
    print('Starting snakemake with:')
    print(args)
    print(kwargs)
    # snakemake.snakemake(*args, printdag=True, **kwargs)
    # snakemake.snakemake(*args, printrulegraph=True, **kwargs)
    # snakemake.snakemake(*args, printfilegraph=True, **kwargs)
    snakemake.snakemake(*args, **kwargs)
    print('Snakemake has ended, press "q" + ENTER to continue')


def run_snakemake(project_path, verbose=False, return_thread=False, **kwargs):

    project_path = os.path.abspath(project_path)
    os.chdir(project_path)

    if 'cores' not in kwargs or kwargs['cores'] is None:
        kwargs['cores'] = os.cpu_count()

    if verbose:
        print(f'kwargs = {kwargs}')

    if return_thread:
        snk_func = snakemake.snakemake
    else:
        snk_func = _snakemake

    # Copy the main snakemake file to the project
    snk_file = os.path.join(project_path, 'snakemake', 'run_main.smk')
    shutil.copy(os.path.join(get_repo_path(), 'snakefiles', 'run_main.smk'), snk_file)

    # Starting snakemake process
    snk_p = Process(
        target=snk_func,
        args=(snk_file,),
        kwargs=kwargs
    )
    snk_p.start()

    if return_thread:
        return snk_p
    else:
        key = ''
        while key.lower() != 'q':
            key = input()

        if snk_p.is_alive():
            print('Terminating Snakemake ...')
            snk_p.terminate()

        # Waiting for snakemake to finish
        snk_p.join()


def _parameter_str_to_dict(params):
    if params is None:
        return dict()
    params = str.split(params, '=')
    param_dict = dict()
    for idx in range(0, len(params), 2):
        try:
            param_dict[params[idx]] = int(params[idx + 1])
        except ValueError:
            param_dict[params[idx]] = float(params[idx + 1])
    return param_dict


def run(
        project_path=None,
        target='membrane_prediction',
        parameters=None,
        roi=None,
        unit='px',
        cores=None,
        gpus=None,
        unlock=False,
        quiet=False,
        dryrun=False,
        return_thread=False,
        cluster=None,
        qos='normal',
        rerun=False,
        restart_times=1,
        debug=False,
        verbose=False
):

    assert cluster in [None, 'slurm']
    if gpus is None:
        if cluster is None:
            gpus = 1
        else:
            gpus = 8

    parameters = _parameter_str_to_dict(parameters)

    # Makes sure the project path is not None and points to a valid project
    project_path = get_current_project_path(project_path=project_path)

    if verbose:
        print(f'project_path = {project_path}')
        print(f'target = {target}')
        print(f'roi = {roi}')
        print(f'unit = {unit}')
        print(f'dryrun = {dryrun}')

    # lock_fp, lock_status = lock_project(project_path=project_path)
    # if lock_status == 'is_locked_error':
    #     print('Project is locked, another instance is already running.')
    #     print('If you are sure that this is not the case, delete the lock file:')
    #     print(f'{lock_fp}')
    #     return 1
    # elif lock_status == 'could_not_lock_error':
    #     print('Could not lock the project. No write permission?')
    #     return 1

    # Create or clean the run requests folder
    if not os.path.exists(os.path.join(project_path, '.run_requests')):
        os.mkdir(os.path.join(project_path, '.run_requests'))
    else:
        from glob import glob
        write_blocks = glob(os.path.join(project_path, '.run_requests', '.request_*'))
        for wb in write_blocks:
            os.remove(wb)

    kwargs = dict(
        resources={
            'gpu': 1
        },
        cores=cores,
        unlock=unlock,
        quiet=quiet,
        dryrun=dryrun,
        force_incomplete=True
    )

    if cluster is not None:
        if not os.path.exists(os.path.join(project_path, 'log')):
            os.mkdir(os.path.join(project_path, 'log'))
        if cluster == 'slurm':
            kwargs['resources']['gpu'] = gpus
            # kwargs['resources']['membrane_prediction_writer'] = 64  # In theory writing in parallel works
            # kwargs['resources']['supervoxels_writer'] = 64
            kwargs['cluster'] = (
                "sbatch "
                "-p {params.p} {params.gres} "
                "-t {resources.time_min} "
                "--mem={resources.mem_mb} "
                "-c {resources.cpus} "
                "-o log/{rule}_{wildcards}d.%N.%j.out "
                "-e log/{rule}_{wildcards}d.%N.%j.err "
                "--mail-type=FAIL "
                f"--mail-user={os.environ['MAIL']} "
                f"-A {os.environ['GROUP']} "
                f"--qos={qos}")
            kwargs['nodes'] = cores
            kwargs['restart_times'] = restart_times
            kwargs['force_incomplete'] = True
        else:
            raise RuntimeError(f'Not supporting cluster = {cluster}')

    if rerun:
        kwargs['forcerun'] = [f'run_{target}']
    # if qos == 'low':
    #     kwargs['force_incomplete'] = True

    # Prepare snakefile(s)
    if target == 'gt_cubes':
        prepare_gt_extract(project_path=project_path, verbose=verbose)
    elif target in ['membrane_prediction', 'supervoxels']:
        prepare_run(
            [target],
            roi=roi,
            unit=unit,
            project_path=project_path,
            debug=debug,
            verbose=verbose)
    elif target[:7] == 'stitch-':
        # This requests the stitching
        assert 'beta' in parameters, 'Supply a value for beta using "--param beta=0.5"'
        base_target = target[7:]
        config_seg = get_config(base_target, project_path=project_path)
        beta_target = f'{base_target}_b{str.replace(str(parameters["beta"]), ".", "_")}'
        if not 'xml_path_stitched' in config_seg['segmentations'][beta_target]:
            init_beta_map(
                f'{beta_target}_stitched',
                base_target,
                parameters['beta'],
                stitched=True,
                project_path=project_path,
                verbose=verbose
            )

        prepare_stitching(
            target,
            parameters['beta'],
            roi=roi,
            unit=unit,
            project_path=project_path,
            verbose=verbose
        )
    else:
        # --- Assuming a segmentation! ---
        # # Make a target list since we need multiple targets here
        # targets = [target]
        # Generate the beta target maps
        beta_targets = []
        config_seg = get_config(target, project_path=project_path)
        betas = config_seg['mc_args']['betas']
        beta_targets.extend([f'{target}_b{str.replace(str(beta), ".", "_")}' for beta in betas])
        # Initialize the beta maps (if they don't exist yet)
        print(beta_targets)
        for idx, bm in enumerate(beta_targets):
            if bm not in config_seg['segmentations'].keys():
                print(f'Initializing beta map: {bm}')
                init_beta_map(
                    bm,
                    target,
                    betas[idx],
                    project_path=project_path,
                    verbose=verbose
                )
            else:
                print(f'Beta map {bm} exists.')
        # Prepare the run
        prepare_run(
            [target],
            roi=roi,
            unit=unit,
            project_path=project_path,
            verbose=verbose
        )

    # Run snakemake
    if return_thread:
        return run_snakemake(project_path, verbose=verbose, return_thread=return_thread, **kwargs)
    else:
        run_snakemake(project_path, verbose=verbose, **kwargs)

    # unlock_project(project_path=project_path)
    return 0


if __name__ == '__main__':

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Starts a run of the CebraEM workflow.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-p', '--project_path', type=str, default=None,
                        help='Path of the project, the current path by default')
    parser.add_argument('-t', '--target', type=str, default='membrane_prediction',
                        help=('Defines the map(s) to compute, can be any of the following:\n'
                              '    "membrane_prediction"\n'
                              '    "supervoxels"\n'
                              '    "[any_segmentation_map_id]"\n'
                              '    "gt_cubes"'))
    parser.add_argument('-par', '--parameters', type=str, default=None,
                        help='Parameters that are fed to the workflow within run.json["misc"]\n'
                             'For example when running stitching, define the beta-map: '
                             'run.py -t stich-seg_map --param beta=0.6')
    parser.add_argument('-r', '--roi', type=float, default=None, nargs=6,
                        metavar=('Z', 'Y', 'X', 'depth', 'height', 'width'),
                        help='Defines a region of interest to which the requested run is confined')
    parser.add_argument('-u', '--unit', type=str, default='px',
                        choices=('px', 'um', 'nm'),
                        help='The unit that defines roi. Can be [px, um, nm]')
    parser.add_argument('-c', '--cores', type=int, default=None,
                        help='Maximum number of CPU cores used, defaults to all available cores')
    parser.add_argument('-g', '--gpus', type=int, default=None,
                        help='Maximum number of GPUs to use, defaults to 1 for local and 8 for cluster computation')
    parser.add_argument('--unlock', action='store_true',
                        help='Use this flag to unlock a locked snakemake workflow')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Do not print any default job information (default False)')
    parser.add_argument('-d', '--dryrun', action='store_true',
                        help='Only dry-run the workflow (default False)')
    parser.add_argument('--cluster', type=str, default=None,
                        help='Enable running jobs on a cluster, currently only Slurm is supported')
    parser.add_argument('--qos', type=str, default='normal',
                        help="Quality of service for cluster jobs ['lowest', 'low', 'normal', 'high', 'highest']")
    parser.add_argument('--rerun', action='store_true',
                        help='Trigger re-running of the respective target chunks')
    parser.add_argument('--restart_times', type=int, default=1,
                        help='How many times a job is restarted if it fails')
    parser.add_argument('--debug', action='store_true',
                        help='Runs in debug mode')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    project_path = args.project_path
    target = args.target
    parameters = args.parameters
    roi = args.roi
    unit = args.unit
    cores = int(args.cores) if args.cores is not None else None
    gpus = args.gpus
    unlock = args.unlock
    quiet = args.quiet
    cluster = args.cluster
    qos = args.qos
    rerun = args.rerun
    restart_times = args.restart_times
    verbose = args.verbose
    dryrun = args.dryrun
    debug = args.debug

    # ----------------------------------------------------
    # Imports

    import os
    import snakemake
    from multiprocessing import Process
    import shutil

    from cebra_em.prepare_snakefiles import prepare_run, prepare_gt_extract, prepare_stitching
    from cebra_em_core.project_utils.project import (
        get_current_project_path,
        lock_project,
        unlock_project
    )
    from cebra_em.misc.repo import get_repo_path
    from cebra_em_core.project_utils.config import get_config
    from cebra_em_core.cebra_em_project import init_beta_map

    # ----------------------------------------------------

    run(
        project_path=project_path,
        target=target,
        parameters=parameters,
        roi=roi,
        unit=unit,
        cores=cores,
        gpus=gpus,
        unlock=unlock,
        quiet=quiet,
        cluster=cluster,
        verbose=verbose,
        dryrun=dryrun,
        qos=qos,
        rerun=rerun,
        restart_times=restart_times,
        debug=debug
    )

