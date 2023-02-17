
import os
from .common.project import make_project_structure
from .common.params import copy_default_params, query_parameters
from .common.config import (
    init_mask_config,
    init_main_config,
    init_raw_config,
    init_image_config,
    get_config,
    get_mask_xml,
    set_version
)
from .common.tasks import compute_task_positions
from .common.dependencies import init_dependencies
from .common.mobie_utils import (
    get_mobie_project_path,
    init_with_raw,
    init_membrane_prediction,
    init_supervoxels,
    init_mask,
)
from .common.version import __version__


def init_parameters(
        raw_params=None,
        mem_params=None,
        sv_params=None,
        mask_params=None,
        project_path=None,
        has_mask=False,
        verbose=False
):
    if has_mask:
        images = ('raw', 'mask', 'membrane_prediction', 'supervoxels')
    else:
        images = ('raw', 'membrane_prediction', 'supervoxels')

    # Copy the default parameters
    copy_default_params(
        params=images,
        project_path=project_path,
        verbose=verbose)

    # Query the parameters for images (mask, membrane_prediction, supervoxels)
    query_parameters(
        dict(
            raw=raw_params,
            membrane_prediction=mem_params,
            supervoxels=sv_params,
            mask=mask_params
        ) if has_mask else dict(
            raw=raw_params,
            membrane_prediction=mem_params,
            supervoxels=sv_params
        ),
        project_path=project_path
    )


def init_configs(
        raw_data_xml,
        mask_xml=None,
        project_path=None,
        version=None,
        force=False,
        verbose=False
):

    has_mask = mask_xml is not None

    init_main_config(project_path=project_path, verbose=verbose)
    init_raw_config(raw_data_xml, project_path=project_path, force=force)
    if has_mask:
        init_mask_config(mask_xml, project_path=project_path, force=force)
    init_image_config('membrane_prediction', project_path=project_path, force=force)
    init_image_config('supervoxels', project_path=project_path, force=force)

    if version is not None:
        set_version(version, project_path=project_path)


def compute_all_task_positions(
        images=('membrane_prediction', 'supervoxels'),
        project_path=None,
        verbose=False
):

    for image in images:
        compute_task_positions(image, project_path=project_path, verbose=verbose)


def init_all_dependencies(
        images=('membrane_prediction', 'supervoxels'),
        project_path=None,
        n_workers=1,
        verbose=False
):

    for image in images:
        init_dependencies(image, project_path=project_path, n_workers=n_workers, verbose=verbose)


def init_mobie_dataset(
        project_path=None,
        verbose=False
):

    mobie_project_path = get_mobie_project_path(project_path=project_path)
    config_raw = get_config('raw', project_path=project_path)
    raw_xml_path = config_raw['xml_path']
    mask_xml_path = get_mask_xml(project_path=project_path)

    # Make the mobie project path
    os.mkdir(mobie_project_path)

    # Initialize the mobie project pointing to the raw data BDV file
    dataset_name = 'CebraINF'
    image_name = 'em-raw'
    dataset_folder, _, _ = init_with_raw(
        mobie_project_path, dataset_name, raw_xml_path, image_name,
        project_path=project_path, verbose=verbose
    )

    # Initialize the mask
    if mask_xml_path is not None:
        if verbose:
            print('Initializing project with mask!')
        init_mask(
            dataset_folder, mask_xml_path, "em-mask",
            project_path=project_path,
            verbose=verbose
        )
    else:
        if verbose:
            print('No mask supplied.')

    # Initialize the membrane prediction
    init_membrane_prediction(
        dataset_name,
        project_path=project_path,
        verbose=verbose
    )

    # Initialize the supervoxels
    init_supervoxels(
        dataset_name,
        project_path=project_path,
        verbose=verbose
    )


def init_project(
        project_path,
        raw_data_xml,
        mask_xml=None,
        raw_params=None,
        mem_params=None,
        sv_params=None,
        mask_params=None,
        max_workers=1,
        force=False,
        verbose=False
):
    """
    Initializes a CebraEM project

    :param project_path: Path of the project to be initialized
    :param raw_data_xml: BDV xml file from the raw data
    :param mask_xml: BDV xml of a segmentation that will be used to mask computations
    :param mem_params: dictionary or json file containing a dictionary with parameters for the membrane prediction
        None (default): the user will be queried for parameters
        "suppress_query": a default set will be used without user query
            (see inf_proj/params/membrane_prediction_defaults.json)
        "some_file.json": This file has to define ALL required parameters
            (see inf_proj/params/membrane_prediction_defaults.json)
    :param sv_params: dictionary or json file containing a dictionary with parameters for supervoxel computation
        None (default): the user will be queried for parameters
        "suppress_query": a default set will be used without user query
            (see inf_proj/params/supervoxels_defaults.json)
        "some_file.json": This file has to define ALL required parameters
            (see inf_proj/params/supervoxels_defaults.json)
    :param mask_params: dictionary or json file containing a dictionary with parameters for the mask
        None (default): the user will be queried for parameters
        "suppress_query": a default set will be used without user query
            (see inf_proj/params/mask_defaults.json)
        "some_file.json": This file has to define ALL required parameters
            (see inf_proj/params/mask_defaults.json)
    :param max_workers: The maximum amount of parallel workers used while setting up the project
    :param force: Force the project creation even if target folder is not empty (use with care!)
    :param verbose:
    :return:
    """

    assert project_path is not None
    project_path = os.path.join(project_path, '')
    if verbose:
        print(f'project_path = {project_path}')

    print('')
    print('Making project structure ...')

    # Generate the project folder structure
    make_project_structure(project_path, ignore_non_empty=force)

    print('')
    print('Setting parameters ...')

    # Copy the default parameters
    # Query the parameters for images (mask, membrane_prediction, supervoxels)
    init_parameters(
        raw_params=raw_params,
        mem_params=mem_params,
        sv_params=sv_params,
        mask_params=mask_params,
        project_path=project_path,
        has_mask=mask_xml is not None,
        verbose=verbose
    )

    print('')
    print('Initializing config files ...')

    # Initialize config files for the project (main) and the initial images
    #  (raw, mask, membrane_prediction, supervoxels)
    init_configs(
        raw_data_xml,
        mask_xml=mask_xml,
        project_path=project_path,
        version=__version__,
        force=force,
        verbose=verbose
    )

    # Initialize the mobie project
    print('')
    print('Initializing mobie project ...')
    init_mobie_dataset(project_path=project_path, verbose=verbose)

    # Compute position grid for each dataset
    print('')
    print('Computing positions ...')
    compute_all_task_positions(project_path=project_path, verbose=verbose)

    # Initialize the dependencies for each image
    print('')
    print('Initializing dependencies ...')
    init_all_dependencies(project_path=project_path, n_workers=max_workers, verbose=verbose)



