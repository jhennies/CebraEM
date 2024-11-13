
import os
import numpy as np


def confine_annotation_to_supervoxel_level(
        input_filepath,
        supervoxel_filepath,
        target_filepath,
        input_key=None,
        supervoxel_key=None,
        crop_to_supervoxels=False
):

    from pybdv.util import open_file

    # Get the segmentation and supervoxels
    with open_file(input_filepath, mode='r') as f:
        seg = f[input_key][:]
    with open_file(supervoxel_filepath, mode='r') as f:
        sv = f[supervoxel_key][:]

    if crop_to_supervoxels:
        from cebra_em_core.dataset.data import crop_center
        seg = crop_center(seg, sv.shape)

    # Transfer the segmentation to the supervoxel level
    new_seg = np.zeros(sv.shape, dtype=seg.dtype)
    for seg_lbl in np.unique(seg):

        if seg_lbl > 0:

            # which svs overlap with this label?
            sv_lbls = np.unique(sv[seg == seg_lbl])

            # Iterate these labels to check if they intersect with more than 50 % of their area
            for sv_lbl in sv_lbls:

                seg_in_sv, counts = np.unique(seg[sv == sv_lbl], return_counts=True)
                id_of_seg_lbl = list(seg_in_sv).index(seg_lbl)
                intersection_ratio = counts[id_of_seg_lbl] / np.sum(counts)

                if intersection_ratio >= 0.5:
                    new_seg[sv == sv_lbl] = seg_lbl

    # Save the result
    from cebra_em_core.dataset.bdv_utils import create_simple_bdv_h5_dataset
    create_simple_bdv_h5_dataset(target_filepath, new_seg, attrs=None)


def import_annotation(
        input_filepath,
        cube_id,
        organelle,
        crop_center=False,
        input_key=None,
        overwrite=False,
        project_path=None,
        verbose=False
):

    # Check if specified gt cube or annotation exists
    from cebra_em_core.project_utils.gt import get_gt_cube_ids, get_gt_dirpath

    valid_gt_ids = get_gt_cube_ids(project_path=project_path)
    if cube_id not in valid_gt_ids:
        print(f'The ground truth for ID = {cube_id} has to exist! Existing IDs are {valid_gt_ids}')
        return

    gt_dirpath = get_gt_dirpath(cube_id, project_path)
    target_filepath = os.path.join(gt_dirpath, f'{organelle}.h5')
    if os.path.exists(target_filepath) and not overwrite:
        print(f'The annotation for {organelle} in cube with ID={id} already exists! Use -o or --overwrite to overwrite.')
        return

    from pybdv.metadata import get_key

    supervoxel_filepath = os.path.join(gt_dirpath, 'sv.h5')

    confine_annotation_to_supervoxel_level(
        input_filepath,
        supervoxel_filepath,
        target_filepath,
        input_key=input_key,
        supervoxel_key=get_key(True, 0, 0, 0),
        crop_to_supervoxels=crop_center
    )

