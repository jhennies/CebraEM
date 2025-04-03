import numpy as np


def load_h5(filepath, binarize=False):
    from h5py import File
    with File(filepath, mode='r') as f:
        data = f['data'][:]
    if binarize:
        data[data > 0] = 255
        data = data.astype('uint8')
    return data


def load_tif(filepath, binarize=False):
    from tifffile import imread
    data = imread(filepath)
    if binarize:
        data[data > 0] = 255
        data = data.astype('uint8')
    return data


def compute_iou(a, b):
    a = a.astype(int)
    a[a > 0] = 1
    b = b.astype(int)
    b[b > 0] = 1
    intersection = np.zeros(a.shape)
    intersection[(a + b) == 2] = 1
    intersection = intersection.sum()

    union = np.zeros(a.shape)
    union[a > 0] = 1
    union[b > 0] = 1
    union = union.sum()

    return intersection / union


def confine_segmentation(seg, ref_seg, dilate=3):
    # Dilate the reference segmentation
    from scipy.ndimage import binary_dilation
    from skimage.morphology import disk
    ref_seg_dil = binary_dilation(ref_seg > 0, structure=disk(dilate)[None, :])
    seg[np.logical_not(ref_seg_dil)] = 0
    return seg


def iou_ann_mib(ann_seg_fp, mib_seg_fp):
    ann_seg = load_h5(ann_seg_fp, binarize=True)
    mib_seg = load_tif(mib_seg_fp, binarize=True)

    iou_full = compute_iou(ann_seg, mib_seg)

    ann_seg = confine_segmentation(ann_seg, mib_seg)
    iou_annotated_mib = compute_iou(ann_seg, mib_seg)

    print(f'iou_full = {iou_full}')
    print(f'iou_annotated_mib = {iou_annotated_mib}')


def rename_file(filename):
    import re
    # Match the relevant parts of the filename
    match = re.match(r'Labels_([^-]+-\d-\d+nm-\d+)-raw_.*?(_[a-z]+)?.tif', filename)

    if match:
        base_name = match.group(1)
        suffix = match.group(2) if match.group(2) else ''
        return f"{base_name}-{suffix[1:]}.h5"

    return None  # Return None if pattern doesn't match


if __name__ == '__main__':

    mib_dirpath = '/media/julian/Data/projects/hennies/cebra-em-publication/cebra-ann-benchmark/iou/mib-segmentations'
    ann_dirpath = '/media/julian/Data/projects/hennies/cebra-em-publication/cebra-ann-benchmark/iou/ann-segmentations'
    from glob import glob
    import os
    mib_filepaths = glob(os.path.join(mib_dirpath, 'Labels*.tif'))

    for mib_fp in mib_filepaths:
        """
        mib_fp = Labels_calu-3-5nm-000-raw_01_241207_2x_34min_er.tif
        ann_fp = calu-3-5nm-000-er.h5
        """
        ann_fp = os.path.join(
            ann_dirpath,
            rename_file(os.path.split(mib_fp)[1])
        )

        print(f'mib_fp = {mib_fp}')
        print(f'ann_fp = {ann_fp}')

        if not os.path.exists(ann_fp):
            print(f'skipping {ann_fp} ... ')

        # iou_ann_mib(ann_fp, mib_fp)

    # # Initial test
    # ann_fp = '/media/julian/Data/projects/hennies/cebra-em-publication/cebra-ann-benchmark/iou/ann-segmentations/hela-1-10nm-015-mito.h5'
    # mib_fp = '/media/julian/Data/projects/hennies/cebra-em-publication/cebra-ann-benchmark/iou/mib-segmentations/Labels_hela-1-10nm-015-raw_01_250103_1x_12min.tif'
    #
    # iou_ann_mib(ann_fp, mib_fp)
