
import numpy as np
import os
from glob import glob
import time
from cebra_em_core.dataset.bdv_utils import BdvDatasetAdvanced


def request_run(idx, path=None, name='n', verbose=False):

    def _request_run(idx, path, name):
        timestamp = time.time()
        open(os.path.join(path, f'.request_{name}_{idx}_{timestamp}'), mode='w').close()
        if verbose:
            print(f'request_run: timestamp = {timestamp}')
            print(f'request_run: idx = {idx}')
        return timestamp

    def _is_queued(idx, path, name, timestamp):

        if verbose:
            print(f'is_queued: timestamp = {timestamp}')
            print(f'is_queued: idx = {idx}')

        # Find all requests
        requests = glob(os.path.join(path, f'.request_{name}_*'))

        found_own_request = False
        queued = False
        if verbose:
            print(f'requests = {requests}')
            print(f'found_own_request = {found_own_request}')
        for request in requests:
            request_idx = int(os.path.split(request)[1].split('_')[-2])
            request_time = float(os.path.split(request)[1].split('_')[-1])
            if request_idx != idx and timestamp > request_time:
                queued = True
            if request_idx == idx:
                found_own_request = True
            if verbose:
                print(f'in loop: request_idx = {request_idx}')
                print(f'found_own_request = {found_own_request}')
        if not found_own_request:
            raise RuntimeError('Request file missing!')

        return queued

    path = os.getcwd() if path is None else path

    timestamp = _request_run(idx, path, name)
    time.sleep(1)
    while _is_queued(idx, path, name, timestamp):
        time.sleep(1)

    return timestamp


def remove_request(idx, timestamp, path=None, name='n'):
    path = os.getcwd() if path is None else path
    os.remove(os.path.join(path, f'.request_{name}_{idx}_{timestamp}'))


def vol_to_bdv(
        volume,
        dataset_path,
        position,
        downscale_mode='mean',
        halo=None,
        background_value=None,
        unique=False,
        update_max_id=False,
        cast_type=None,
        block_description=None,
        verbose=False
):
    if block_description is None:
        block_description = dict(
            path='./',
            idx=0,
            name='n',
        )

    # Type casting
    if cast_type is not None:
        volume = volume.astype(cast_type)

    # Getting location within the Bdv dataset
    if halo is not None:
        position_halo = np.array(position) - np.array(halo)
    else:
        position_halo = np.array(position)
    s_ = np.s_[
         position_halo[0]: position_halo[0] + volume.shape[0],
         position_halo[1]: position_halo[1] + volume.shape[1],
         position_halo[2]: position_halo[2] + volume.shape[2]
         ]

    # Initialize the bdv dataset
    bdv_ds = BdvDatasetAdvanced(
        path=dataset_path,
        timepoint=0,
        setup_id=0,
        downscale_mode=downscale_mode,
        halo=halo,
        background_value=background_value,
        unique=unique,
        update_max_id=update_max_id,
        n_threads=1,
        verbose=verbose
    )

    # Block from parallel writing
    idx, path, name = block_description['idx'], block_description['path'], block_description['name']
    ts = request_run(idx, path=os.path.join(path, f'.run_requests'), name=name, verbose=verbose)
    # Write the data
    bdv_ds[s_] = volume
    # Remove block
    remove_request(idx, ts, path=os.path.join(path, f'.run_requests'), name=name)

