
def link_gt(
    cube_ids,
    organelle,
    image_name,
    val=False,
    project_path=None,
    verbose=False
):

    if verbose:
        print(f'cube_ids = {cube_ids}')
        print(f'organelle = {organelle}')
        print(f'image_name = {image_name}')
        print(f'val = {val}')
        print(f'project_path = {project_path}')

    from cebra_em_core.project_utils.gt import link_gt_cubes
    link_gt_cubes(
        cube_ids, organelle, image_name,
        val=val,
        project_path=project_path,
        verbose=verbose
    )


if __name__ == '__main__':

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Link ground truth cubes to a segmentation.\n\n'
                    'Requirements:\n'
                    '  - Initialized ground truth cube(s):  `init_gt.py [args]`\n'
                    '  - Initialized segmentation:          `init_segmentation.py [args]` \n\n'
                    'Next step:\n'
                    '  - Train and run the segmentation:    `run.py -t [segmentation_id]`\n',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('cube_ids', type=int, nargs='+',
                        help='Numeric ids of the ground truth cubes')
    parser.add_argument('organelle', type=str,
                        help='Name of the organelle, must match organelle name of CebraANN export')
    parser.add_argument('image_name', type=str,
                        help='Name of the image to link to')
    parser.add_argument('--val', action='store_true',
                        help='Cube is added as training data by default, set this flag to add it as validation data')
    parser.add_argument('-p', '--project_path', type=str, default=None,
                        help='Path of the project')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    cube_ids = args.cube_ids
    organelle = args.organelle
    image_name = args.image_name
    val = args.val
    project_path = args.project_path
    verbose = args.verbose

    # ----------------------------------------------------

    link_gt(
        cube_ids,
        organelle,
        image_name,
        val=val,
        project_path=project_path,
        verbose=verbose
    )
