
import numpy as np
from vigra.analysis import labelMultiArrayWithBackground
from pybdv.util import open_file, get_key
from pybdv.metadata import get_data_path
import pickle
import json
from cebra_em_core.segmentation.stitching_utils import largest_non_zero_overlap, merge_mappings

from cebra_em_core.dataset.bdv_utils import is_h5
from cebra_em.run_utils.run_specs import get_run_json
from cebra_em_core.project_utils.project import get_current_project_path
from cebra_em_core.project_utils.config import get_config, absolute_path
from cebra_em_core.dataset.data import load_with_zero_padding


def match_ids_at_block_faces(block_faces):

    def _match_ids_at_block_face(bf):

        bf_in = bf[1]
        bfr_in = bf[0]

        bf_in_rel = labelMultiArrayWithBackground(bf_in.astype('uint32'), background_value=0)
        bfr_in_rel = labelMultiArrayWithBackground(bfr_in.astype('uint32'), background_value=0)

        # print(f'np.unique(bf_in) = {np.unique(bf_in)}')
        # print(f'np.unique(bfr_in) = {np.unique(bfr_in)}')
        # print(f'np.unique(bf_in_rel) = {np.unique(bf_in_rel)}')
        # print(f'np.unique(bfr_in_rel) = {np.unique(bfr_in_rel)}')

        map = {}
        for lbl_rel in np.unique(bf_in_rel):
            if lbl_rel > 0:
                ref_lbl_rel, overlap_ratio = largest_non_zero_overlap(bf_in_rel, bfr_in_rel, lbl_rel)
                if ref_lbl_rel is not None and overlap_ratio > 0.5:
                    lbl = int(bf_in[np.where(bf_in_rel == lbl_rel)].mean())
                    ref_lbl = int(bfr_in[np.where(bfr_in_rel == ref_lbl_rel)].mean())
                    if lbl in map:
                        map[lbl].append(ref_lbl)
                    else:
                        map[lbl] = [ref_lbl]

        return map

    a = [_match_ids_at_block_face(bf) for bf in block_faces]
    if verbose:
        print(f'ids at block face = {a}')
    b = merge_mappings(a)
    if verbose:
        print(f'merged mappings = {b}')

    return b


def load_faces(image_xml, pos, shp):
    """
    This function finds the left, top and front block face from the input location and shape and the respective
    block-faces from the adjacent cubes
    Returns an array of shape = (3, 2, m, n) where m x n is the size of the block-face
     - dim0: the three block-faces (left, top and front)
     - dim1: 0: the block-face of the adjacent chunk; 1: the block-face of this chunk
        (the order has practical reasons, see the code at comment 'block-face definitions')
     - dim2 and dim3: the block face images
    """

    # Block-face definitions
    faces_s = [
        [  # left
            [pos[0], pos[1], pos[2] - 1],
            [pos[0] + shp[0], pos[1] + shp[1], pos[2] + 1],
            [shp[0], shp[1], 2]
        ],
        [  # top
            [pos[0], pos[1] - 1, pos[2]],
            [pos[0] + shp[0], pos[1] + 1, pos[2] + shp[2]],
            [shp[0], 2, shp[2]]
        ],
        [  # front
            [pos[0] - 1, pos[1], pos[2]],
            [pos[0] + 1, pos[1] + shp[1], pos[2] + shp[2]],
            [2, shp[1], shp[2]]
        ]
    ]

    # Load from the dataset
    data_path = get_data_path(image_xml, return_absolute_path=True)

    def _load(d, starts, ends, shape, idx):
        data = load_with_zero_padding(d, starts, ends, shape)
        if idx == 0:
            data = np.moveaxis(data, 2, 0)
        if idx == 1:
            data = np.moveaxis(data, 1, 0)
        return data

    key = get_key(is_h5(image_xml), timepoint=0, setup_id=0, scale=0)
    with open_file(data_path, mode='r') as ds:
        faces = [_load(ds[key], s[0], s[1], s[2], idx) for idx, s in enumerate(faces_s)]

    return faces


def find_mapping(data_path, pos, shp):

    # Load the left, top and front block-faces with the data from the respective adjacent chunks
    block_faces = load_faces(data_path, pos, shp)

    # Find the mapping
    return match_ids_at_block_faces(block_faces)


if __name__ == '__main__':

    print(f">>> STARTING: find_mapping for {snakemake.params['image_name']}[{snakemake.wildcards['idx']}]")

    project_path = get_current_project_path(None)
    verbose = get_run_json(project_path=project_path)['verbose']
    beta = get_run_json(project_path=project_path)['misc']['beta']

    image_name = snakemake.params['image_name']
    idx = int(snakemake.wildcards['idx'])

    if verbose:
        print(f'image_name = {image_name}')
        print(f'idx = {idx}')

    config_img = get_config(image_name, project_path=project_path)
    img_xml_rel_path = config_img['segmentations'][f'{image_name}_b{str.replace(str(beta), ".", "_")}']['xml_path']
    img_xml_path = absolute_path(img_xml_rel_path, project_path=project_path)

    positions_fp = absolute_path(config_img['positions'], project_path=project_path)
    with open(positions_fp, 'rb') as f:
        position = pickle.load(f)[idx]
    batch_shape = config_img['batch_shape']

    mapping = find_mapping(img_xml_path, position, batch_shape)

    with open(snakemake.output[0], 'w') as f:
        json.dump(mapping, f)

    print(f"<<< DONE: find_mapping for {snakemake.params['image_name']}[{snakemake.wildcards['idx']}]")
