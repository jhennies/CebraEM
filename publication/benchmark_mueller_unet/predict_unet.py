import os.path


def predict_unet(
        input_filepath,
        output_dirpath,
        model_dirpath,
        unet_src_dirpath=None,
        verbose=False
):

    if verbose:
        print(f'input_filepath = {input_filepath}')
        print(f'output_dirpath = {output_dirpath}')

    if unet_src_dirpath is not None:
        import sys
        sys.path.append(unet_src_dirpath)

    import numpy as np
    from tifffile import imread, imwrite
    from csbdeep.utils import Path
    from csbdeep.utils.tf import limit_gpu_memory
    limit_gpu_memory(fraction=0.8, total_memory=12000)
    from model import UNetConfig, UNet
    np.random.seed(42)
    from glob import glob

    def apply(model, x0):
        x = x0.astype(np.float32) / 255.
        n_tiles = tuple(int(np.ceil(s / 196)) for s in x0.shape)
        y_full = model.predict(x, axes="ZYX", normalizer=None, n_tiles=n_tiles)

        # y = y_full >= 0.5
        y = (y_full * 255).astype('uint8')

        return y

    filepaths = glob(input_filepath)

    model_basedir, model_name = os.path.split(model_dirpath)
    if verbose:
        print(f'model_name = {model_name}')
        print(f'model_basedir = {model_basedir}')
    model = UNet(None, model_name, basedir=model_basedir)

    for filepath in filepaths:

        filepath = Path(filepath)

        # load file
        x0 = imread(filepath)

        y = apply(model, x0)

        # save output
        out = Path(output_dirpath)

        out.mkdir(exist_ok=True, parents=True)
        imwrite(out / f"{Path(filepath).stem}.unet.tif", y)  # .astype(np.uint16))


if __name__ == '__main__':

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description=('Predicts a Mueller et al. U-Net\n'
                     'Derived from https://github.com/betaseg/protocol-notebooks/blob/main/unet/run_unet.ipynb'),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_filepath', type=str,
                        help='Path to a 3D tif image file; Can be a glob for multiple files. e.g. "/path/to/*.tif"')
    parser.add_argument('output_dirpath', type=str,
                        help='Folder where the result will be written to')
    parser.add_argument('model_dirpath', type=str,
                        help='Folder which contains the model (output of train_unet.py)')
    parser.add_argument('-src', '--unet_src_dirpath', type=str, default=None,
                        help='Local location of https://github.com/betaseg/protocol-notebooks/tree/main/unet')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    input_filepath = args.input_filepath
    output_dirpath = args.output_dirpath
    model_dirpath = args.model_dirpath
    unet_src_dirpath = args.unet_src_dirpath
    verbose = args.verbose

    predict_unet(
        input_filepath,
        output_dirpath,
        model_dirpath,
        unet_src_dirpath=unet_src_dirpath,
        verbose=verbose,
    )

