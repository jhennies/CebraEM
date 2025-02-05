
def create_prediction_data(
        input_dirpath,
        output_dirpath,
        verbose=False
):

    if verbose:
        print(f'input_dirpath = {input_dirpath}')
        print(f'output_dirpath = {output_dirpath}')

    import os.path
    from glob import glob
    from h5py import File
    from tifffile import imwrite

    filepaths = glob(os.path.join(input_dirpath, '*-raw.h5'))

    for filepath in filepaths:

        out_filepath = os.path.join(
            output_dirpath,
            os.path.splitext(os.path.split(filepath)[1])[0] + '.tif'
        )

        with File(filepath, mode='r') as f:
            data = f['data'][:]

        imwrite(out_filepath, data, imagej=True, metadata={"axes": "ZYX"})


if __name__ == '__main__':

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Converts raw em hdf5 files to 3D tifs',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_dirpath', type=str, default=None,
                        help='Folder containing the raw em hdf5 files')
    parser.add_argument('output_dirpath', type=str,
                        help='Target folder')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    input_dirpath = args.input_dirpath
    output_dirpath = args.output_dirpath
    verbose = args.verbose

    create_prediction_data(
        input_dirpath,
        output_dirpath,
        verbose=verbose
    )
