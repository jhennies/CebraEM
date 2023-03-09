
import numpy as np
from elf.segmentation.workflows import FEATURE_NAMES
from vigra.analysis import labelVolume

from cebra_em_core.segmentation.elf_utils import predict_node_classification_mc_wf, node_classification_mc_wf


def predict_mc_wf_classifiers(
        input_vols,  # np.array([raw, mem, sv, mask])
        rf_filepath,
        nrf_filepath,
        mask=None,
        n_threads=1,
        feature_names=FEATURE_NAMES,
        resolution=(1., 1., 1.),
        verbose=False
):

    data_raw = input_vols[0]
    data_mem = input_vols[1]
    data_sv = input_vols[2]
    data_mask = mask

    assert data_raw.shape == data_mem.shape == data_sv.shape, \
        f'raw.shape = {data_raw.shape}, mem.shape = {data_mem.shape}, sv.shape = {data_sv.shape}'

    shape = data_raw.shape
    print(f'shape = {shape}')

    return_nothing = False
    if np.min(data_raw) == np.max(data_raw):
        return_nothing = True
    if np.min(data_mem) == np.max(data_mem):
        return_nothing = True
    if np.max(data_sv) == 0:
        return_nothing = True
    if np.min(data_sv) == np.max(data_sv):
        return_nothing = True
    if not np.logical_and(data_mem, data_sv).max():
        # The non-zero regions of membrane prediction and supervoxels don't intersect
        return_nothing = True
    if return_nothing:
        return np.zeros(shape)

    print('Relabeling supervoxels ...')
    data_sv = labelVolume(data_sv.astype('uint32') + 1).astype('uint32')

    # There must be no zero label in the supervoxel map
    print(f'0 in data_sv = {0 in data_sv}')
    if 0 in data_sv:
        data_sv += 1
        print(f'0 in data_sv = {0 in data_sv}')

    if verbose:
        print(f'data_raw.dtype = {data_raw.dtype}')
        print(f'data_mem.dtype = {data_mem.dtype}')
        print(f'data_sv.dtype = {data_sv.dtype}')
        print(f'data_raw.shape = {data_raw.shape}')

    print('Predicting classifiers for node classification multicut ...')

    res_dict = predict_node_classification_mc_wf(
        raw=data_raw,
        boundaries=data_mem,
        watershed=data_sv,
        rf=rf_filepath,
        nrf=nrf_filepath,
        mask=data_mask,
        feature_names=feature_names,
        rel_resolution=resolution,
        n_threads=n_threads
    )

    return res_dict


def multicut_from_predicted(
        in_dict,  # Dictionary as returned by predict_mc_wf_classifiers
        beta,
        verbose=False
):

    if verbose:
        print('multicut_from_predicted()')
        print(in_dict.keys())
        print(in_dict['result'].keys())

    # Fetch inputs

    mc_out = in_dict['result']
    watershed = mc_out['watershed']
    edge_probs = mc_out['edge_probs']
    node_probs = mc_out['node_probs']
    edge_sizes = mc_out['edge_sizes']
    bounds = in_dict['bounds']
    mask = in_dict['mask']
    shape = in_dict['shape']

    # Run the multicut

    multicut_solver = 'kernighan-lin'
    solver_kwargs = {}

    seg = node_classification_mc_wf(
        watershed,
        edge_probs,
        node_probs,
        edge_sizes,
        multicut_solver,
        beta,
        beta_nodes=None,
        mask=mask,
        solver_kwargs=solver_kwargs,
        n_threads=1
    )

    # Pad the result

    if verbose:
        print(f'seg.shape = {seg.shape}')
        print(f'np.unique(seg) = {np.unique(seg)}')

    if verbose:
        print(f'Padding result to original size ...')
    # Pad the result to match the input shape
    vol_out = np.zeros(shape, seg.dtype)
    vol_out[bounds] = seg
    if verbose:
        print(f'vol_out.shape = {vol_out.shape}')
        print(f'np.unique(vol_out) = {np.unique(vol_out)}')

    if verbose:
        print(f'Removing everything outside the mask ...')
    # Remove everything outside the mask

    if verbose:
        print(f'np.unique(vol_out) = {np.unique(vol_out)}')

    return vol_out