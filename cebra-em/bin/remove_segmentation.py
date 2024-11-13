
def main():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Removes a segmentation map\n'
                    'To print existing segmentation datasets and their linked ground truth cubes use: `log_gt -d`',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('segmentation_name', type=str,
                        help='Name of the segmentation map to delete; Use the full name, e.g. "mito_iter01"')
    parser.add_argument('-p', '--project_path', type=str, default=None,
                        help='Path of the project')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    segmentation_name = args.segmentation_name
    project_path = args.project_path
    debug = args.debug
    verbose = args.verbose

    # ----------------------------------------------------

    from cebra_em_core.cebra_em_project import remove_segmentation
    remove_segmentation(
        segmentation_name,
        project_path=project_path,
        debug=debug,
        verbose=verbose
    )
