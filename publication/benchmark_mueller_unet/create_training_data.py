
"""
FIXME: The below should make individual training folders for each organelle as well (not only for each dataset)

[For each dataset: (This is done outside of this script)]

    Find masks -> masks = []
    Determine organelles from masks -> organelles = [...]
    For each organelle:
        Find images and masks for the organelle -> images = [...], masks = [...]
        if split condition is met -> split_images = True else False
        if split_images:
            do the split such that newlen(images) = len(images) * (num_splits+1)
        Set up dictionaries to link images/masks to filenames -> dict_images = {image[0]: fn[0], ...}

        For each image:
            Load from hdf5 -> image_data = np.array()
            if not split_images:
                Determine output location -> out_filepath = .../dataset-organelle-name/type/images/image-basename.tif
                Save image as 3D tif to out_filepath
            else:
                Split image in half -> image[0] = image[:n/2], image[1] = image[n/2:]
                for idx in [0, 1]:
                    Determine output location -> out_filepath = .../dataset-organelle-name/type/images/image-basename-idx.tif
                    Save image[idx] as 3D tif to out_filepath
        For each mask:
            Convert mask to uint8
            Set values > 0 to 255
            if not split_images:
                Determine output location -> out_filepath = .../dataset-organelle-name/type/masks/mask-basename.tif
                Save mask as 3D tif to out_filepath
            else:
                Split mask in half -> mask[0] = mask[:n/2], mask[1] = mask[n/2:]
                for idx in [0, 1]:
                    Determine output location -> out_filepath = .../dataset-organelle-name/type/masks/mask-basename-idx.tif
                    Save mask[idx] as 3D tif to out_filepath

"""
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

    filepaths = glob(os.path.join(dirpath, '*.h5'))
    contents = [os.path.splitext(filepath)[0].split('-')[-1] for filepath in filepaths]

    mask_filepaths = []
    raw_filepaths = []
    for idx, content in enumerate(contents):
        if content == organelle:
            mask_filepaths.append(filepaths[idx])
            cube_name = os.path.split(filepaths[idx])[1].split(f'-{organelle}')[0]
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


def _process_organelle_set(parts_info, target_dirpaths):

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

        raw_out_filepath = os.path.join(
            raw_out_dirpath,
            os.path.splitext(os.path.split(raw_filepath)[1])[0] + f'-{start_at}.tif'
        )
        imwrite(raw_out_filepath, raw_data, imagej=True, metadata={"axes": "ZYX"})

        with File(mask_filepath, mode='r') as f:
            mask_data = f['data'][start_at: start_at + length, :]

        mask_data[mask_data > 0] = 255
        mask_data = mask_data.astype('uint8')

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
        _process_organelle_set(parts_info, target_dirpaths[organelle])

    print('------------------------------------------------')


if __name__ == '__main__':

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Initializes a CebraEM project',
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
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    input_gt_dirpath = args.input_gt_dirpath
    output_dirpath = args.output_dirpath
    min_val_fraction = args.min_val_fraction
    allow_splits = args.allow_splits
    organelles = args.organelles
    verbose = args.verbose

    create_training_data(
        input_gt_dirpath,
        output_dirpath,
        min_val_fraction=min_val_fraction,
        allow_splits=allow_splits,
        organelles=organelles,
        verbose=verbose
    )
