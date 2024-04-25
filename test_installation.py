import numpy as np
import os


def test_installation():

    from h5py import File

    # Load test data
    with File('test_data/input.h5', mode='r') as f:
        input_data = f['data'][:]

    from cebra_em_core.bioimageio.cebra_net import run_cebra_net
    mem = np.array(run_cebra_net(input_data))
    # print(mem)
    # print(f'mem.shape = {mem.shape}')
    # print(f'mem.dtype = {mem.dtype}')

    print('Membrane prediction successfully computed...')

    from vigra.analysis import labelVolumeWithBackground
    # labelVolumeWithBackground(np.zeros((64, 64, 64), dtype='float32'))
    # labelVolumeWithBackground(mem)
    from cebra_em_core.segmentation.supervoxels import watershed_dt_with_probs
    sv = watershed_dt_with_probs(mem)
    # print(f'sv.shape = {sv.shape}')

    print(f'Supervoxels successfully computed ...')

    out_filepath = 'test_data/tmp_output.h5'
    if os.path.exists(out_filepath):
        os.remove(out_filepath)
    with File(out_filepath, mode='w') as f:
        f.create_dataset('mem', data=mem, compression='gzip')
        f.create_dataset('sv', data=sv.astype('uint16'), compression='gzip')

    print('\nAll tests successful!')


if __name__ == '__main__':
    test_installation()
