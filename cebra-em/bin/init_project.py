

if __name__ == '__main__':

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Initializes a CebraEM project\n\n'
                    'Next steps:\n'
                    '  - Run membrane prediction and supervoxels:  `run.py -t supervoxels`\n'
                    '  - Initialize a segmentation:                `init_segmentation.py [args]`\n'
                    '  - Initialize ground truth cubes:            `init_gt.py [args]`\n',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('project_path', type=str, default=None,
                        help='Path of the project to be initialized')
    parser.add_argument('raw_data_xml', type=str,
                        help='BDV xml file from the raw data')
    parser.add_argument('-m', '--mask_xml', type=str, default=None,
                        help='BDV xml of a segmentation that will be used to mask computations')
    parser.add_argument('--general_params', type=str, default=None,
                        help=('Json file containing a dictionary with set of parameters used for the maps below '
                              'if nothing else is specified\n'
                              '    None (default): the user will be queried for parameters\n'
                              '    "suppress_query": a default set will be used without user query '
                              '(see project_utils/params/general_defaults.json)'))
    parser.add_argument('--mem_params', type=str, default=None,
                        help=('Json file containing a dictionary with parameters for the membrane prediction\n'
                              '    None (default): the user will be queried for parameters\n'
                              '    "suppress_query": a default set will be used without user query '
                              '(see project_utils/params/membrane_prediction_defaults.json)'))
    parser.add_argument('--sv_params', type=str, default=None,
                        help=('Json file containing a dictionary with parameters for supervoxel computation\n'
                              '    None (default): the user will be queried for parameters\n'
                              '    "suppress_query": a default set will be used without user query '
                              '(see project_utils/params/supervoxels_defaults.json)'))
    parser.add_argument('--mask_params', type=str, default=None,
                        help=('Json file containing a dictionary with parameters for the mask\n'
                              '    None (default): the user will be queried for parameters\n'
                              '    "suppress_query": a default set will be used without user query '
                              '(see project_utils/params/mask_defaults.json)'))
    parser.add_argument('--max_workers', type=int, default=1,
                        help='The maximum amount of parallel workers used while setting up the project')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force the project creation even if target folder is not empty (use with care!)')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    project_path = args.project_path
    raw_data_xml = args.raw_data_xml
    mask_xml = args.mask_xml
    general_params = args.general_params
    mem_params = args.mem_params
    sv_params = args.sv_params
    mask_params = args.mask_params
    max_workers = args.max_workers
    force = args.force
    verbose = args.verbose

    # ----------------------------------------------------

    from cebra_em_core import init_project

    init_project(
        project_path,
        raw_data_xml,
        mask_xml=mask_xml,
        general_params=general_params,
        mem_params=mem_params,
        sv_params=sv_params,
        mask_params=mask_params,
        max_workers=max_workers,
        force=force,
        verbose=verbose
    )
