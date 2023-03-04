
import numpy as np
import pickle

from elf.segmentation.workflows import DEFAULT_RF_KWARGS
from elf.segmentation.workflows import FEATURE_NAMES
from elf.io import open_file
from vigra.analysis import labelVolume

from cebra_em_core.segmentation.elf_utils import DEFAULT_NRF_KWARGS
from cebra_em_core.segmentation.elf_utils import edge_and_node_training

from cebra_em_core.project_utils.config import get_config
from cebra_em.run_utils.run_specs import get_run_json


def crop_center(vol, shape):

    vol_shape = np.array(vol.shape)
    shape = np.array(shape)

    if np.abs(vol_shape - shape).max() != 0:

        start = ((vol_shape - shape) / 2).astype('int')
        vol = vol[
              start[0]: start[0] + shape[0],
              start[1]: start[1] + shape[1],
              start[2]: start[2] + shape[2]
        ]

    assert np.abs(np.array(vol.shape) - shape).max() == 0

    return vol


def get_rf_model(
        output_filepaths,
        input_raw_files,
        input_mem_files,
        input_sv_files,
        input_gt_files,
        learning_kwargs=DEFAULT_RF_KWARGS,
        node_learning_kwargs=DEFAULT_NRF_KWARGS,
        feature_names=None
):

    feature_names = FEATURE_NAMES if feature_names is None else feature_names

    assert type(input_raw_files) == list

    n = len(input_raw_files)
    assert len(input_mem_files) == n
    assert len(input_sv_files) == n
    assert len(input_gt_files) == n

    raw_train = []
    mem_train = []
    gt_train = []
    sv_train = []

    for i in range(n):

        # ground truth
        filename = input_gt_files[i]
        with open_file(filename, 'r') as f:
            gt = f['data'][:].astype(np.float32)
            gt_shape = gt.shape
            gt_train.append(gt)

        # raw data
        filename = input_raw_files[i]
        with open_file(filename, 'r') as f:
            raw = f['data'][:].astype(np.float32)
            raw = crop_center(raw, gt_shape)
            raw_train.append(raw)

        # membrane prediction -- boundaries
        filename = input_mem_files[i]
        with open_file(filename, 'r') as f:
            mem = f['data'][:].astype(np.float32)
            mem = crop_center(mem, gt_shape)
            mem_train.append(mem)

        # supervoxels
        filename = input_sv_files[i]
        with open_file(filename, 'r') as f:
            sv = f['data'][:].astype('uint64')
            sv = crop_center(sv, gt_shape)
            sv = labelVolume(sv.astype('uint32')).astype('uint16')
            sv_train.append(sv)

    if 'n_jobs' in learning_kwargs:
        n_thr = learning_kwargs['n_jobs']
    else:
        n_thr = None

    rf = edge_and_node_training(
        raw=raw_train, boundaries=mem_train, labels=gt_train, watershed=sv_train,
        learning_kwargs=learning_kwargs,
        node_learning_kwargs=node_learning_kwargs,
        feature_names=feature_names,
        n_threads=n_thr
    )

    nrf = rf[1]
    rf = rf[0]
    with open(output_filepaths[1], 'wb') as f:
        print(output_filepaths[1])
        print(f'dumping nrf as type {type(nrf)}')
        pickle.dump(nrf, f)
    with open(output_filepaths[0], 'wb') as f:
        print(output_filepaths[0])
        print(f'dumping rf as type {type(rf)}')
        pickle.dump(rf, f)


if __name__ == '__main__':

    image = snakemake.params['image_name']

    print(f">>> STARTING: Training for {image}")

    verbose = get_run_json(project_path=None)

    config_seg = get_config(image)
    learning_kwargs = config_seg['learning_kwargs']
    node_learning_kwargs = config_seg['node_learning_kwargs']

    mc_args = config_seg['mc_args']
    if 'feature_names' in mc_args:
        feature_names = mc_args['feature_names']
    else:
        feature_names = None

    output_filepaths = snakemake.output

    n_gt = int(len(snakemake.input) / 4)
    input_gt_files = snakemake.input[:n_gt]
    inputs = np.reshape(snakemake.input[n_gt:], (n_gt, 3))
    input_raw_files = list(inputs[:, 0])
    input_mem_files = list(inputs[:, 1])
    input_sv_files = list(inputs[:, 2])
    print('INPUTS')
    print(input_raw_files)
    print(input_mem_files)
    print(input_sv_files)
    print(input_gt_files)

    get_rf_model(
        output_filepaths=output_filepaths,
        input_raw_files=input_raw_files,
        input_mem_files=input_mem_files,
        input_sv_files=input_sv_files,
        input_gt_files=input_gt_files,
        learning_kwargs=learning_kwargs,
        node_learning_kwargs=node_learning_kwargs,
        feature_names=feature_names
    )

    print(f"<<< DONE: Training for {image}")

