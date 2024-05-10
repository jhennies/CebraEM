
def init_gt(
    project_path=None,
    shape=(256, 256, 256),
    position=None,
    bdv_position=None,
    no_padding=False,
    verbose=False
):

    assert position is not None or bdv_position is not None

    if verbose:
        print(f'project_path = {project_path}')
        print(f'shape = {shape}')
        print(f'position = {position}')
        print(f'bdv_position = {bdv_position}')
        print(f'no_padding = {no_padding}')

    # Initialize the ground truth cube
    from cebra_em_core.project_utils.gt import init_gt_cube
    init_gt_cube(
        project_path=project_path,
        shape=shape,
        position=position,
        bdv_position=bdv_position,
        no_padding=no_padding,
        verbose=verbose
    )


def main():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Initializes a ground truth cube at a certain position in the dataset.\n'
                    'To print existing ground truth cube information use: `log_gt`',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-p', '--project_path', type=str, default=None,
                        help='Path of the project')
    parser.add_argument('-s', '--shape', type=int, nargs=3, default=(256, 256, 256),
                        metavar=('Z', 'Y', 'X'),
                        help='Shape of the ground truth cube; default = (256, 256, 256)')
    parser.add_argument('--position', type=int, nargs=3, default=None,
                        metavar=('Z', 'Y', 'X'),
                        help='Position in dataset unit, normally micrometer')
    parser.add_argument('-b', '--bdv_position', type=str, default=None,
                        help=('Position in BDV log format:\n'
                              '   \'{"position":[0.0,0.0,0.0],...}\' or \'(0.0,0.0,0.0)\' or ...\n'
                              '   -> basically any format that contains three floats like this: '
                              '\'(x.xxxx, y.yyyy, z.zzzz)\'\n'))
    parser.add_argument('--no_padding', action='store_true',
                        help='Switch off padding of raw data')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    project_path = args.project_path
    shape = args.shape
    position = args.position
    bdv_position = args.bdv_position
    no_padding = args.no_padding
    verbose = args.verbose

    # ----------------------------------------------------

    init_gt(
        project_path=project_path,
        shape=shape,
        position=position,
        bdv_position=bdv_position,
        no_padding=no_padding,
        verbose=verbose
    )
