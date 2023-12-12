import click
import os
import shutil
import snakemake

from multiprocessing import Process
from cebra_em.prepare_snakefiles import prepare_run, prepare_gt_extract, prepare_stitching
from cebra_em_core.project_utils.project import (
    get_current_project_path,
    lock_project,
    unlock_project
)
from cebra_em.misc.repo import get_repo_path
from cebra_em_core.project_utils.config import get_config
from cebra_em_core.cebra_em_project import init_beta_map
from cebra_em_core.misc.cliutils import execution_options, project_path_option, quiet_option, verbose_option


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


@click.command()
@project_path_option()
@click.option(
    "-t", "--target",
    type=str,
    default="membrane_prediction",
    help="Defines the map(s) to compute"
)
@click.option(
    "--parameter",
    type=str,
    multiple=True,
    help="Parameters that are fed to the workflow within run.json['misc']"
)
@click.option(
    "--roi",
    type=int,
    nargs=6,
    metavar=('Z', 'Y', 'X', 'depth', 'height', 'width'),
    help="Defines a region of interest to which the requested run is confined",
)
@click.option(
    "--unit",
    type=click.Choice(["px", "um", "nm"]),
    default="px",
    help="The unit that defines roi"
)
@execution_options
@click.option(
    "--unlock",
    type=bool,
    is_flag=True,
    help="Use this flag to unlock a locked snakemake workflow"
)
@quiet_option
@click.option(
    "--dryrun",
    type=bool,
    is_flag=True,
    help="Only dry-run the workflow"
)
@click.option(
    "--rerun",
    type=bool,
    is_flag=True,
    help="Trigger re-running of the respective target chunks"
)
@click.option(
    "--restart-times",
    type=click.IntRange(min=0),
    default=1,
    help="How many times a job is restarted if it fails"
)
@verbose_option
def run(
    project_path=None,
    target=None,
    parameter=None,
    roi=None,
    unit=None,
    cores=None,
    gpus=None,
    unlock=None,
    quiet=None,
    dryrun=None,
    cluster=None,
    qos=None,
    rerun=None,
    restart_times=None,
    verbose=None,
    return_thread=False # What is this about?
):
    "Run the CebraEM pipeline workflow. "

    assert cluster in [None, 'slurm']
    if gpus is None:
        if cluster is None:
            gpus = 1
        else:
            gpus = 8

    parameters = {}
    for param in parameter:
        parameters.update(**_parameter_str_to_dict(param))

    # Makes sure the project path is not None and points to a valid project
    project_path = get_current_project_path(project_path=project_path)

    if verbose:
        print(f'project_path = {project_path}')
        print(f'target = {target}')
        print(f'roi = {roi}')
        print(f'unit = {unit}')
        print(f'dryrun = {dryrun}')

    lock_fp, lock_status = lock_project(project_path=project_path)
    if lock_status == 'is_locked_error':
        print('Project is locked, another instance is already running.')
        print('If you are sure that this is not the case, delete the lock file:')
        print(f'{lock_fp}')
        return 1
    elif lock_status == 'could_not_lock_error':
        print('Could not lock the project. No write permission?')
        return 1

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
        dryrun=dryrun
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
        else:
            raise RuntimeError(f'Not supporting cluster = {cluster}')

    if rerun:
        kwargs['forcerun'] = [f'run_{target}']
    if qos == 'low':
        kwargs['force_incomplete'] = True

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

    unlock_project(project_path=project_path)
    return 0


if __name__ == '__main__':
    run()
