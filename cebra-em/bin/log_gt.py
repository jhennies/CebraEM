
def log_gt(project_path=None, datasets=False):
    """
    Shows the available ground truth cubes and their status

    :param project_path: Path of the project
    :param datasets: Show from the dataset's perspective

    :return:
    """
    if datasets:
        from cebra_em_core.project_utils.gt import log_datasets as log_func
    else:
        from cebra_em_core.project_utils.gt import log_gt_cubes as log_func

    log_func(project_path=project_path)


if __name__ == '__main__':

    # ___________________________________________________
    # Command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description='Shows the available ground truth cubes and their status',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-p', '--project_path', type=str, default=None,
                        help='Path of the project')
    parser.add_argument('-d', '--datasets', action='store_true',
                        help="Show from the dataset's perspective")

    args = parser.parse_args()
    project_path = args.project_path
    datasets = args.datasets

    # ___________________________________________________

    log_gt(project_path=project_path, datasets=datasets)
