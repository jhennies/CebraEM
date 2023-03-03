
if __name__ == '__main__':

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Initializes a segmentation',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('organelle', type=str,
                        help='Name of the target organelle')
    parser.add_argument('suffix', type=str,
                        help='This string is appended to form the segmentation image ID. \n'
                             'For example:\n'
                             '    organelle="mito"\n'
                             '    suffix="iter01"\n'
                             'Yields segmentation_id = "mito_iter01"')
    parser.add_argument('--params', type=str, default=None,
                        help=('Json file containing a dictionary with parameters for the membrane prediction\n'
                              '    None (default): the user will be queried for parameters\n'
                              '    "suppress_query": a default set will be used without user query '
                              '(see project_utils/params/membrane_prediction_defaults.json)'))
    parser.add_argument('-p', '--project_path', type=str, default=None,
                        help='Path of the project')
    parser.add_argument('--max_workers', type=int, default=1,
                        help='The maximum amount of parallel workers used while setting up the segmentation')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    organelle = args.organelle
    suffix = args.suffix
    params = args.params
    project_path = args.project_path
    max_workers = args.max_workers
    verbose = args.verbose

    # ----------------------------------------------------

    from cebra_em_core.cebra_em_project import init_segmentation
    init_segmentation(
        organelle, suffix,
        params=params,
        project_path=project_path,
        max_workers=max_workers,
        verbose=verbose
    )
