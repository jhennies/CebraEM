
import numpy as np
from vigra import analysis


def load_with_zero_padding(dataset, starts, ends, shape, verbose=False):

    if np.array(ends).min() < 0:
        # This happens if the data is completely outside of the target dataset (< 0)
        return np.zeros(shape, dtype=dataset.dtype)

    starts_target = np.zeros((3,), dtype=int)
    starts_target[starts < 0] = -starts[starts < 0]
    starts_source = np.zeros((3,), dtype=int)
    starts_source[starts > 0] = starts[starts > 0]

    too_large = (np.array(dataset.shape) - ends) < 0
    ends_target = shape.copy()
    ends_target[too_large] = (np.array(dataset.shape) - starts)[too_large]
    ends_source = ends.copy()
    ends_source[too_large] = np.array(dataset.shape)[too_large]

    if np.array(ends_target).min() < 0:
        # This happens if the data is completely outside of the target dataset (> shape)
        return np.zeros(shape, dtype=dataset.dtype)

    if verbose:
        print(f'shape = {shape}')
        print(f'dataset.shape = {dataset.shape}')
        print(f'starts = {starts}')
        print(f'ends = {ends}')
        print(f'starts_source = {starts_source}')
        print(f'starts_target = {starts_target}')
        print(f'ends_source = {ends_source}')
        print(f'ends_target = {ends_target}')

    slicing_source = np.s_[
        starts_source[0]: ends_source[0],
        starts_source[1]: ends_source[1],
        starts_source[2]: ends_source[2]
    ]
    slicing_target = np.s_[
        starts_target[0]: ends_target[0],
        starts_target[1]: ends_target[1],
        starts_target[2]: ends_target[2]
    ]

    vol = np.zeros(shape, dtype=dataset.dtype)
    vol[slicing_target] = dataset[slicing_source]

    return vol


def get_non_fully_contained_ids(volume):

    vol = volume.copy()
    vol[1:-1, 1:-1, 1:-1] = 0

    return np.unique(vol)[1:]


def relabel_from_mapping(vol, mapping):
    print(np.unique(vol))
    print(mapping.keys())
    shp = vol.shape
    # Flatten
    vol = np.reshape(vol, np.product(shp))
    # Do the mapping
    vol = np.array([mapping[val] for val in vol])
    # Back to original shape
    return np.reshape(vol, shp)


def relabel_with_skip_ids(vol, skip_ids):
    """
    Relabels a volume and skips a list of ids
    The zero label is kept as it is and treated as background
    """

    labels = np.unique(vol)
    labels = labels[np.nonzero(labels)].tolist()
    new_labels = []
    c = 0
    for _ in labels:
        c += 1
        while c in skip_ids:
            c += 1
        new_labels.append(c)

    assert len(labels) == len(new_labels)

    relabel_dict = dict(zip(labels, new_labels))
    relabel_dict[0] = 0

    vol = relabel_from_mapping(vol, relabel_dict)

    return vol


def iou(segmentation, gt):
    segmentation = segmentation.astype(int)
    gt = gt.astype(int)
    # print(f'unique_seg = {np.unique(segmentation)}')
    # print(f'unique_ref = {np.unique(gt)}')
    # print(f'unique_seg_ref = {np.unique(segmentation + gt)}')
    intersection = np.zeros(segmentation.shape)
    intersection[(segmentation + gt) == 2] = 1
    intersection = intersection.sum()
    # print(f'intersection = {intersection}')

    union = np.zeros(segmentation.shape)
    union[segmentation > 0] = 1
    union[gt > 0] = 1
    union = union.sum()
    # print(f'union = {union}')

    return intersection / union


def largest_non_zero_overlap(vol, ref, lbl):
    v, c = np.unique(ref[vol == lbl], return_counts=True)
    biggest_overlap = None
    ratio = None
    for idx in np.argsort(c)[::-1]:
        if v[idx] != 0:
            biggest_overlap = v[idx]
            count = c[idx]
            ratio = count / (ref == biggest_overlap).sum()
            break

    return biggest_overlap, ratio


def assert_list(val):
    if type(val) == np.ndarray:
        val = [int(x) for x in val]
    if type(val) != list:
        return [val]
    else:
        return val


def merge_mappings(mappings, convert_items=None):

    print(f'mappings = {mappings}')
    mapping = {}

    for m in mappings:
        for k, v in m.items():
            v = assert_list(v)
            if convert_items is not None:
                if convert_items == 'int':
                    k = int(k)
                    v = [int(val) for val in v]
                else:
                    raise ValueError('Possible values for convert_items: "int"')

            # print(f'mapping = {mapping}')
            # print(f'k = {k}')
            # print(f'v = {v}')
            # print('-------------')
            if k in mapping.keys():
                mapping[k] = assert_list(np.unique(assert_list(mapping[k]) + v))
            else:
                mapping[k] = v

    return mapping


def match_ids_at_block_faces(block_faces, block_faces_ref, crop=False):

    def _normalize_block_face_shapes(block_face, block_face_ref):
        # Make the block faces match if cropping is enabled, otherwise just check for match
        bf_in = block_face
        bfr_in = block_face_ref
        bf_shp = np.array(block_face.shape)
        bfr_shp = np.array(block_face_ref.shape)
        print(f'bf_shp = {bf_shp}')
        print(f'bfr_shp = {bfr_shp}')
        if (bf_shp - bfr_shp).max() > 0:
            assert (bf_shp - bfr_shp).min() >= 0, \
                'Cropping either block_face or block_face_ref, not both in different dimensions'
            start_pos = ((bf_shp - bfr_shp) / 2).astype(int)
            bf_in = bf_in[
                start_pos[0]: start_pos[0] + bfr_shp[0],
                start_pos[1]: start_pos[1] + bfr_shp[1]
            ]
        elif (bf_shp - bfr_shp).min() < 0:
            assert (bf_shp - bfr_shp).max() <= 0, \
                'Cropping either block_face or block_face_ref, not both in different dimensions'
            start_pos = ((bfr_shp - bf_shp) / 2).astype(int)
            bfr_in = bfr_in[
                 start_pos[0]: start_pos[0] + bf_shp[0],
                 start_pos[1]: start_pos[1] + bf_shp[1]
            ]

        return bf_in, bfr_in

    def _connected_components_with_mapping(im):

        dtype = im.dtype
        cc = analysis.labelImageWithBackground(im.astype('float32')).astype(dtype)

        mapping = {}
        for lbl in np.unique(cc):
            if lbl > 0:
                mapping[int(lbl)] = im[cc == lbl][0]

        return cc, mapping

    def _get_largest_iou(im, ref, lbl):

        v = np.unique(ref[im == lbl])

        largest_iou = 0.
        ref_lbl = None
        for this_lbl in v:
            if this_lbl != 0:
                this_iou = iou(im == lbl, ref == this_lbl)
                if this_iou > largest_iou:
                    largest_iou = this_iou
                    ref_lbl = this_lbl

        return ref_lbl, largest_iou

    def _match_ids_at_block_face_with_iou(block_face, block_face_ref):

        bf = block_face
        bfr = block_face_ref

        if crop:
            bf, bfr = _normalize_block_face_shapes(bf, bfr)

        bf_shp = np.array(bf.shape)
        bfr_shp = np.array(bfr.shape)

        assert np.abs(bf_shp - bfr_shp).max() == 0, f'Block face shapes do not match: {bf_shp}, {bfr_shp}'

        bf_cc, bf_cc_map = _connected_components_with_mapping(bf)
        bfr_cc, bfr_cc_map = _connected_components_with_mapping(bfr)

        mappings = []

        for lbl in np.unique(bf_cc):
            if lbl > 0:
                ref_lbl, largest_iou = _get_largest_iou(bf_cc, bfr_cc, lbl)
                if ref_lbl is not None and largest_iou > 0.5:
                    mappings.append({int(bf_cc_map[int(lbl)]): int(bfr_cc_map[int(ref_lbl)])})

        print(f'mappings = {mappings}')
        return merge_mappings(mappings)

    def _match_ids_at_block_face(block_face, block_face_ref):

        bf_in = block_face
        bfr_in = block_face_ref

        if crop:
            bf_in, bfr_in = _normalize_block_face_shapes(bf_in, bfr_in)

        bf_shp = np.array(block_face.shape)
        bfr_shp = np.array(block_face_ref.shape)

        assert np.abs(bf_shp - bfr_shp).max() == 0, f'Block face shapes do not match: {bf_shp}, {bfr_shp}'

        map = {}
        for lbl in np.unique(bf_in):
            if lbl > 0:
                ref_lbl, overlap_ratio = largest_non_zero_overlap(bf_in, bfr_in, lbl)
                if ref_lbl is not None and overlap_ratio > 0.5:
                    map[int(lbl)] = int(ref_lbl)

        return map

    a = [_match_ids_at_block_face_with_iou(bf, block_faces_ref[idx]) for idx, bf in enumerate(block_faces)]
    print(f'ids at block face = {a}')
    b = merge_mappings(a)
    print(f'merged mappings = {b}')

    return b


def get_block_faces(volume):

    assert volume.ndim == 3, 'Implemented for 3D volumes'

    block_faces = [
        volume[0, :, :],
        volume[-1, :, :],
        volume[:, 0, :],
        volume[:, -1, :],
        volume[:, :, 0],
        volume[:, :, -1]
    ]

    return block_faces


def solve_mapping(mapping, verbose=False):

    solution = mapping

    def _solve_item(k, v):

        new_v = []

        for val in v:
            if val in mapping.keys():
                new_v.extend(assert_list(mapping[val]))
            else:
                new_v.append(val)

        new_v = assert_list(np.unique(new_v))
        if k in new_v:
            new_v.remove(k)
            # print(f'Removed k = {k} from values')
        solution[k] = new_v

        # Returning whether something has changed
        return v == new_v

    solved = False
    it = 0
    while not solved:
        print(f'Iteration = {it}')

        it += 1

        mapping = solution.copy()
        solved = True

        prev_solution = solution.copy()

        for k, v in mapping.items():

            # Returns True if all dependencies within the values of the item are solved
            if not _solve_item(k, v):
                solved = False

            else:
                # If the item still maps to more than one other items a new entry is necessary to finally solve this
                if len(v) > 1:
                    solution[v[0]] = v[1:]
                    solved = False

        if solution == prev_solution:
            print("Warning: Mapping probably not solved but no change detected! Aborting...")
            solved = True

    if verbose:
        print(f'Solution for mapping after {it} iterations:')

    return solution
