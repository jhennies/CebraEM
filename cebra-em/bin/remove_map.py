
def main():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Removes a segmentation map\n'
                    'To print existing segmentation datasets and their linked ground truth cubes use: `log_gt -d`',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('map_name', type=str,
                        help='Name of the segmentation map to delete; Use the full name, e.g. "mito_iter01"')
    parser.add_argument('-p', '--project_path', type=str, default=None,
                        help='Path of the project')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    map_name = args.map_name
    project_path = args.project_path
    verbose = args.verbose

    # ----------------------------------------------------

    from cebra_em_core.cebra_em_project import remove_map
    remove_map(
        map_name,
        project_path=project_path,
        verbose=verbose
    )
