
import os
import numpy as np


def find_bounding_boxes(segmentation, ids=None, resolution=(1., 1., 1.), verbose=False):

    # Get all non-zero voxel coordinates and their corresponding instance IDs
    coords = np.argwhere(segmentation > 0)
    instance_ids = segmentation[coords[:, 0], coords[:, 1], coords[:, 2]]

    # Find unique instance IDs
    if ids is None:
        ids = np.unique(instance_ids)

    bounding_boxes = []

    # Find the bounding box for each unique instance ID
    for idx, instance_id in enumerate(ids):
        if verbose:
            print(f'step = {idx} / {len(ids)}')
            print(f'instance_id = {instance_id}')

        # Get indices of all points belonging to the current instance
        indices = np.where(instance_ids == instance_id)[0]

        # Get the coordinates of these points
        instance_coords = coords[indices]

        # Calculate the bounding box
        z_min, y_min, x_min = instance_coords.min(axis=0)
        z_max, y_max, x_max = instance_coords.max(axis=0)
        anchor_z, anchor_y, anchor_x = instance_coords.mean(axis=0)

        # Store the bounding box
        bounding_boxes.append(dict(
            label_id=instance_id,
            bb_min_x=x_min * resolution[2],
            bb_min_y=y_min * resolution[1],
            bb_min_z=z_min * resolution[0],
            bb_max_x=x_max * resolution[2],
            bb_max_y=y_max * resolution[1],
            bb_max_z=z_max * resolution[0],
            anchor_x=anchor_x * resolution[2],
            anchor_y=anchor_y * resolution[1],
            anchor_z=anchor_z * resolution[0],
            # num_voxels=len(instance_coords)
        ))

    return bounding_boxes


def update_labels_table(
        target,
        initial_downsample_level=3,
        final_downsample_level=1,
        project_path=None,
        cores=os.cpu_count(),
        verbose=False
):

    from cebra_em_core.dataset.data import load_full_downsample_level
    from cebra_em_core.project_utils.config import get_config, absolute_path, relative_path
    from pybdv.metadata import get_data_path, get_key
    from cebra_em_core.dataset.bdv_utils import get_resolution
    from cebra_em_core.dataset.bdv_utils import is_h5

    # Load the data for the instance search
    if target == 'mask':
        from cebra_em_core.project_utils.config import get_mask_xml
        target_xml_path = get_mask_xml(project_path=project_path)
    else:
        from cebra_em_core.project_utils.config import get_segmentation_xml
        target_xml_path = get_segmentation_xml(target, project_path=project_path)
    target_xml_path_abs = absolute_path(target_xml_path, project_path=project_path)
    target_path = get_data_path(target_xml_path_abs, return_absolute_path=True)
    internal_path = get_key(is_h5(target_xml_path_abs), 0, 0, initial_downsample_level)
    resolution = get_resolution(target_xml_path_abs, setup_id=0, downsample_level=initial_downsample_level)
    if verbose:
        print(f'target_xml_path_abs = {target_xml_path_abs}')
        print(f'target_path = {target_path}')
        print(f'internal_path = {internal_path}')
        print(f'resolution = {resolution}')

    init_segmentation = load_full_downsample_level(
        target_path,
        internal_path
    )
    if verbose:
        print(f'init_segmentation.shape = {init_segmentation.shape}')

    # Find the bounding boxes based on the initial downsample level
    if verbose:
        print('Finding bounding boxes ...')
    ids = None
    if target == 'mask':
        config_target = get_config(target, project_path)
        ids = config_target['args']['ids']
    bounding_boxes = find_bounding_boxes(init_segmentation, ids=ids, resolution=resolution, verbose=verbose)

    if verbose:
        print('bounding boxes:')
        print(bounding_boxes)

    # TODO: Refine the bounding boxes

    # Create the labels table
    if verbose:
        print('Creating labels table ...')
    from cebra_em_core.dataset.mobie_utils import create_labels_table
    from cebra_em_core.dataset.mobie_utils import get_mobie_project_path
    mobie_project_dirpath = get_mobie_project_path(project_path=project_path)
    labels_table_filepath = os.path.join(
        mobie_project_dirpath,
        f'{target}.csv'
    )
    create_labels_table(labels_table_filepath, bounding_boxes)

    # Update the mobie table
    if verbose:
        print(f'Updating mobie table ...')
    from cebra_em_core.dataset.mobie_utils import update_mobie_table_entry, get_mobie_table_path
    update_mobie_table_entry(
        get_mobie_table_path(project_path),
        ['labels_table', relative_path(labels_table_filepath, project_path)],
        ['uri', target_xml_path]
    )
