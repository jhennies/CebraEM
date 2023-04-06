
import os
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from vigra.analysis import relabelConsecutive, labelMultiArray
from elf.segmentation.utils import make_3d_edges
from cebra_em_core.dataset.data import small_objects_to_zero


def crop_center(data, target_shape):
    shp = np.array(data.shape)
    target_shp = np.array(target_shape)

    start = ((shp - target_shape) / 2).astype('int')

    return data[
           start[0]: start[0] + target_shp[0],
           start[1]: start[1] + target_shp[1],
           start[2]: start[2] + target_shp[2]
    ]


def crop_to_same_shape(*volumes):

    shapes = [vol.shape for vol in volumes]
    min_shape = np.min(shapes, axis=0)

    volumes = [crop_center(vol, min_shape) for vol in volumes]

    return volumes


def merge_small_segments(m, size=48, verbose=False):

    from scipy.ndimage import minimum_filter, maximum_filter

    print('Small objects to zero ...')
    m = small_objects_to_zero(m + 1, size, verbose=verbose)

    if verbose:
        print('Filling the holes ...')
        print(f'm.shape = {m.shape}')
        print(f'm.dtype = {m.dtype}')
    # m = watershedsNew(m.astype('float32'), seeds=m.astype('uint32'), neighborhood=26)[0] - 1
    while 0 in m:
        m_ = maximum_filter(m, 3)
        m[m == 0] = m_[m == 0]
    if verbose:
        print('... done!')

    return m


def _split_into_maps(joint_map, single_maps):

    new_maps = {}
    for idx in np.unique(joint_map):
        max_overlap = 0
        assign_to = ''
        for name, data in single_maps.items():
            if name not in new_maps.keys():
                new_maps[name] = np.zeros(joint_map.shape, dtype='uint64')
            v, c = np.unique(data[joint_map == idx], return_counts=True)
            v = v[np.argsort(c)]
            c = c[np.argsort(c)]
            if v[-1] == 0:
                if len(v) > 1:
                    this_max_overlap = c[-2]
                else:
                    this_max_overlap = 0
            else:
                this_max_overlap = c[-1]
            if max_overlap < this_max_overlap:
                assign_to = name
                max_overlap = this_max_overlap
        new_maps[assign_to][joint_map == idx] = idx

    return new_maps


def transfer_seg_to_sv_level(seg, sv, original_seg):

    # # Version4: Only question the supervoxels where there was a change
    boundaries = make_3d_edges(seg)
    boundaries_original = make_3d_edges(original_seg)
    changed_sv = np.unique(sv[boundaries != boundaries_original])
    len_sv = len(changed_sv)
    new_seg = deepcopy(seg)
    print(f'len(sv) = {len_sv}')

    def _assign_svs(idx):
        v, c = np.unique(seg[sv == idx], return_counts=True)
        v = v[np.argsort(c)]
        new_seg[sv == idx] = v[-1]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as tpe:
        tasks = [
            tpe.submit(
                _assign_svs, sv_lbl
            )
            for sv_lbl in changed_sv
        ]
        [t.result() for t in tasks]

    return new_seg


def segmentation_cleanup(
        layer_data, sv,
        merge_small=False, fill_holes=False, conn_comp=False,
        merge_small_size=48,
        verbose=False
):

    if verbose:
        print(f'merge_small_size = {merge_small_size}')
        print(f'type(merge_small_size) = {type(merge_small_size)}')

    # Make joint instance segmentation
    m = deepcopy(layer_data['instances'])
    for name, data in layer_data.items():
        if name != 'instances':
            max_val = m.max()
            m[data > 0] = deepcopy(data[data > 0] + max_val)

    m_ = deepcopy(m)
    m = m.astype('float32')

    # TODO: Fill holes
    if fill_holes:
        print('Warning: hole filling not implemented yet')

    # Connected component analysis
    if conn_comp:
        m = labelMultiArray(m)
        sv_dtype = sv.dtype
        sv = labelMultiArray(sv.astype('uint32')).astype(sv_dtype)

    # Merge small segments
    if merge_small:
        m = merge_small_segments(m, size=merge_small_size, verbose=verbose)

    # Always relabel (FIXME needed?)
    # m = relabel_consecutive(m, sort_by_size=False) + 1
    m = relabelConsecutive(m.astype('uint32'))[0] + 1
    # Assert supervoxel level
    m = transfer_seg_to_sv_level(m, sv, m_)

    m = m.astype('uint64')

    # Split the result into the input layer maps
    layer_data = _split_into_maps(m, layer_data)

    return layer_data, sv
