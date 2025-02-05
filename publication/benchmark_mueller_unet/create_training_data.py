
import os.path
from glob import glob
import numpy as np


def _determine_organelles(dirpath):

    filepaths = glob(os.path.join(dirpath, '*.h5'))
    organelles = [os.path.splitext(filepath)[0].split('-')[-1] for filepath in filepaths]
    organelles = [organelle for organelle in organelles if organelle not in ['raw', 'pad']]

    return np.unique(organelles).tolist()


def _make_target_folder_structure(target_dirpath, organelles):

    if not os.path.exists(target_dirpath):
        os.mkdir(target_dirpath)

    def _add_dirpath(root_dirpath, pathname):
        dirpath = os.path.join(root_dirpath, pathname)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        return dirpath

    def _add_images_and_masks_dirpaths(root_dirpath):
        return(
            _add_dirpath(root_dirpath, 'images'),
            _add_dirpath(root_dirpath, 'masks')
        )

    def _make_organelle_folder_structure(organelle_dirpath):
        if not os.path.exists(organelle_dirpath):
            os.mkdir(organelle_dirpath)

        train_dirpath = _add_dirpath(organelle_dirpath, 'train')
        train_images_dirpath, train_masks_dirpath = _add_images_and_masks_dirpaths(train_dirpath)
        val_dirpath = _add_dirpath(organelle_dirpath, 'val')
        val_images_dirpath, val_masks_dirpath = _add_images_and_masks_dirpaths(val_dirpath)

        return dict(
            train_dirpath=train_dirpath,
            train_images_dirpath=train_images_dirpath,
            train_masks_dirpath=train_masks_dirpath,
            val_dirpath=val_dirpath,
            val_images_dirpath=val_images_dirpath,
            val_masks_dirpath=val_masks_dirpath
        )

    path_dict = dict()
    for organelle in organelles:
        path_dict[organelle] = _make_organelle_folder_structure(os.path.join(target_dirpath, organelle))

    return path_dict


def _get_image_and_mask_filepaths(dirpath, organelle):

    mask_filepaths = glob(os.path.join(dirpath, f'*-{organelle}.h5'))

    raw_filepaths = []
    for filepath in mask_filepaths:
        cube_name = os.path.split(filepath)[1].split(f'-{organelle}')[0]
        raw_filepaths.append(os.path.join(dirpath, f'{cube_name}-raw.h5'))

    return mask_filepaths, raw_filepaths


def _determine_splits(number_of_maps, max_number_of_splits, min_val_fraction=0.25):

    from math import ceil
    max_tolerance = 0.1
    val_target_amount = number_of_maps * min_val_fraction
    ceil_val_target_amount = ceil(val_target_amount)
    if (ceil_val_target_amount - val_target_amount) / number_of_maps <= max_tolerance:
        return ceil_val_target_amount, False

    ceil_val_target_amount = ceil(val_target_amount * (max_number_of_splits + 1)) / (max_number_of_splits + 1)
    return ceil_val_target_amount, not ceil_val_target_amount.is_integer()


def _determine_val_cubes(mask_filepaths, raw_filepaths, number_of_val_cubes):

    # image_filepaths = np.array(range(5))
    # number_of_val_cubes = 1.5

    try:
        cubes_to_parts_factor = int(1 / (number_of_val_cubes - int(number_of_val_cubes)))
    except ZeroDivisionError:
        cubes_to_parts_factor = 1

    total_number_of_parts = len(mask_filepaths) * cubes_to_parts_factor
    number_of_val_parts = int(number_of_val_cubes * cubes_to_parts_factor)

    val_ids = np.array(range(int(total_number_of_parts / number_of_val_parts / 2), total_number_of_parts, int(total_number_of_parts / number_of_val_parts)))

    parts_mask_filepaths = [fp for fp in mask_filepaths for _ in range(cubes_to_parts_factor)]
    parts_raw_filepaths = [fp for fp in raw_filepaths for _ in range(cubes_to_parts_factor)]

    parts_info = []
    for idx in range(0, total_number_of_parts, cubes_to_parts_factor):
        keep_separated = False
        for j in range(cubes_to_parts_factor):
            if idx + j in val_ids:
                keep_separated = True

        if keep_separated:
            for j in range(cubes_to_parts_factor):
                parts_info.append([
                    parts_mask_filepaths[idx + j],
                    parts_raw_filepaths[idx + j],
                    j / cubes_to_parts_factor,
                    1 / cubes_to_parts_factor,
                    idx + j in val_ids
                ])
        else:
            parts_info.append([
                parts_mask_filepaths[idx],
                parts_raw_filepaths[idx],
                0.0,
                1.0,
                idx in val_ids
            ])

    return parts_info


def _bin_volume(volume, bin_factor):
    """Bins a 3D volume by an integer factor using NumPy reshaping and averaging."""
    assert volume.shape[0] % bin_factor == 0
    assert volume.shape[1] % bin_factor == 0
    assert volume.shape[2] % bin_factor == 0

    shape = (volume.shape[0] // bin_factor, bin_factor,
             volume.shape[1] // bin_factor, bin_factor,
             volume.shape[2] // bin_factor, bin_factor)

    return volume.reshape(shape).mean(axis=(1, 3, 5))


def _process_organelle_set(parts_info, target_dirpaths, binning=1):

    from h5py import File
    from tifffile import imwrite

    for info in parts_info:

        mask_filepath = info[0]
        raw_filepath = info[1]
        start_at = info[2]
        length = info[3]
        is_val = info[4]

        if is_val:
            raw_out_dirpath = target_dirpaths['val_images_dirpath']
            mask_out_dirpath = target_dirpaths['val_masks_dirpath']
        else:
            raw_out_dirpath = target_dirpaths['train_images_dirpath']
            mask_out_dirpath = target_dirpaths['train_masks_dirpath']
        with File(raw_filepath, mode='r') as f:
            raw_data = f['data']

            start_at = int(raw_data.shape[0] * start_at)
            length = int(raw_data.shape[0] * length)

            raw_data = raw_data[start_at: start_at + length, :]

        if binning > 1:
            raw_data = _bin_volume(raw_data, binning)

        raw_out_filepath = os.path.join(
            raw_out_dirpath,
            os.path.splitext(os.path.split(raw_filepath)[1])[0] + f'-{start_at}.tif'
        )
        imwrite(raw_out_filepath, raw_data, imagej=True, metadata={"axes": "ZYX"})

        with File(mask_filepath, mode='r') as f:
            mask_data = f['data'][start_at: start_at + length, :]

        mask_data[mask_data > 0] = 255
        mask_data = mask_data.astype('uint8')

        if binning > 1:
            mask_data = _bin_volume(mask_data, binning)

        mask_out_filepath = os.path.join(
            mask_out_dirpath,
            os.path.splitext(os.path.split(mask_filepath)[1])[0] + f'-{start_at}.tif'
        )
        imwrite(mask_out_filepath, mask_data, imagej=True, metadata={"axes": "ZYX"})


def create_training_data(
        input_gt_dirpath,
        output_dirpath,
        min_val_fraction=0.25,
        allow_splits=1,
        organelles=None,
        binning=1,
        verbose=False
):

    if verbose:
        print(f'input_gt_dirpath = {input_gt_dirpath}')
        print(f'output_dirpath = {output_dirpath}')
        print(f'min_val_fraction = {min_val_fraction}')
        print(f'allow_splits = {allow_splits}')
        print(f'organelles = {organelles}')

    if organelles is None:
        organelles = _determine_organelles(input_gt_dirpath)
    if verbose:
        print(f'organelles = {organelles}')

    target_dirpaths = _make_target_folder_structure(output_dirpath, organelles)
    if verbose:
        print(f'target_dirpaths = {target_dirpaths}')

    for organelle in organelles:

        if verbose:
            print('------------------------------------------------')
            print(f'organelle = {organelle}')

        # Find images and masks for the organelle
        mask_filepaths, raw_filepaths = _get_image_and_mask_filepaths(input_gt_dirpath, organelle)
        if verbose:
            print(f'mask_filepaths = {mask_filepaths}')
            print(f'raw_filepaths = {raw_filepaths}')

        # Determine how to split the dataset
        # -> parts_info = [mask_filepath, raw_filepath, start_in_cube, fraction_of_cube, is_val]
        val_amount, do_splits = _determine_splits(len(mask_filepaths), allow_splits, min_val_fraction=min_val_fraction)
        parts_info = _determine_val_cubes(mask_filepaths, raw_filepaths, val_amount)
        if verbose:
            print(f'parts_info = {parts_info}')

        # Process the dataset
        _process_organelle_set(parts_info, target_dirpaths[organelle], binning=binning)

    print('------------------------------------------------')


if __name__ == '__main__':

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Creates the training data structure required by the Mueller et al. U-Net',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_gt_dirpath', type=str, default=None,
                        help='Folder containing the ground truth data')
    parser.add_argument('output_dirpath', type=str,
                        help='Target folder which contains the Mueller et al. U-Net input structure')
    parser.add_argument('-minval', '--min_val_fraction', type=float, default=0.25,
                        help='The minimum fraction of data that will be used for validation; default=0.25')
    parser.add_argument('-splits', '--allow_splits', type=int, default=1,
                        help='Allow splitting data cubes n times; default=1')
    parser.add_argument('--organelles', type=str, nargs='+', default=None,
                        help='If set, only the specified organelle(s) will be used; Default=None')
    parser.add_argument('--binning', type=int, default=1,
                        help='Binning factor used to downsample the data; [1, 2, 4, 8, ...]; default=1 (no binning)')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    input_gt_dirpath = args.input_gt_dirpath
    output_dirpath = args.output_dirpath
    min_val_fraction = args.min_val_fraction
    allow_splits = args.allow_splits
    organelles = args.organelles
    binning = args.binning
    verbose = args.verbose

    create_training_data(
        input_gt_dirpath,
        output_dirpath,
        min_val_fraction=min_val_fraction,
        allow_splits=allow_splits,
        organelles=organelles,
        binning=binning,
        verbose=verbose
    )
