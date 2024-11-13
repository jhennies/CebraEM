
def main():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Imports an annotation for a ground truth cube.\n'
                    'Note that the cube needs to be properly extracted using the CebraEM functionality, '
                    'the annotation can then be done on any software and imported using this function.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_filepath', type=str,
                        help='Input file containing the annotation. For only supporting hdf5')
    parser.add_argument('cube_id', type=int,
                        help='The ground truth cube ID to which the annotation is assigned')
    parser.add_argument('organelle', type=str,
                        help='Name of the organelle which is annotated')
    parser.add_argument('--crop_center', action='store_true',
                        help='Crop the center of the annotation according to the shape of the ground truth cube')
    parser.add_argument('-key', '--input_key', type=str, default=None,
                        help='Internal path of the input file; default=None, looks for the first entry it finds')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Overwrites an existing annotation. Make sure it is the correct one!')
    parser.add_argument('-p', '--project_path', type=str, default=None,
                        help='Path of the project')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    input_filepath = args.input_filepath
    cube_id = args.cube_id
    crop_center = args.crop_center
    organelle = args.organelle
    input_key = args.input_key
    overwrite = args.overwrite
    project_path = args.project_path
    verbose = args.verbose

    # ----------------------------------------------------

    from cebra_em_core.segmentation.gt_import import import_annotation
    import_annotation(
        input_filepath,
        cube_id,
        organelle,
        crop_center=crop_center,
        input_key=input_key,
        overwrite=overwrite,
        project_path=project_path,
        verbose=verbose
    )
