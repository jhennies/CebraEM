
def main():
    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            'Update the MoBIE table anchors such that objects in a segmentation can be navigated to'
            'by clicking the respective table entry.'
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('target', type=str,
                        help=('The segmentation map to compute, can be any of the following:\n'
                              '    "[any_segmentation_map_id]"\n'
                              '    "mask"'))
    parser.add_argument('-init', '--initial_downsample_level', type=int, default=3,
                        help='The downsample level at which objects are detected; '
                             'The lower the quicker, however very small items might be missed')
    parser.add_argument('-final', '--final_downsample_level', type=int, default=1,
                        help='The downsample level to refine the properties of each instance; '
                             'If -init equals -final, no refinement is performed')
    parser.add_argument('-p', '--project_path', type=str, default=None,
                        help='Path of the project, the current path by default')
    parser.add_argument('-c', '--cores', type=int, default=None,
                        help='Maximum number of CPU cores used, defaults to all available cores')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    target = args.target
    initial_downsample_level = args.initial_downsample_level
    final_downsample_level = args.final_downsample_level
    project_path = args.project_path
    cores = args.cores
    verbose = args.verbose

    from cebra_em_core.segmentation.instances import update_mobie_table

    update_mobie_table(
        target,
        initial_downsample_level=initial_downsample_level,
        final_downsample_level=final_downsample_level,
        project_path=project_path,
        cores=cores,
        verbose=verbose
    )
