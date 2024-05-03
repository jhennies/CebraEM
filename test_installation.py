import numpy as np
import os


def test_installation():

    from h5py import File

    out_fp = 'test_data/tmp_test'
    if not os.path.exists(out_fp):
        os.mkdir(out_fp)

    # Load test data
    with File('test_data/raw.h5', mode='r') as f:
        input_data = f['data'][:]

    from cebra_em_core.bioimageio.cebra_net import run_cebra_net
    mem = np.array(run_cebra_net(input_data))
    # print(mem)
    # print(f'mem.shape = {mem.shape}')
    # print(f'mem.dtype = {mem.dtype}')

    print('\n-----------------------------------------------')
    print('Membrane prediction successfully computed...')
    print('-----------------------------------------------\n')

    from cebra_em_core.segmentation.supervoxels import watershed_dt_with_probs
    sv = watershed_dt_with_probs(mem)
    # print(f'sv.shape = {sv.shape}')

    print('\n-----------------------------------------------')
    print(f'Supervoxels successfully computed ...')
    print('-----------------------------------------------\n')

    with File('test_data/mito.h5', mode='r') as f:
        labels = f['data'][:]

    from cebra_em_core.segmentation.elf_utils import edge_and_node_training
    rf = edge_and_node_training(
        raw=[input_data.astype('float32')],
        boundaries=[mem.astype('float32')],
        labels=[labels.astype('float32')],
        watershed=[sv]
    )
    import pickle
    with open(os.path.join(out_fp, 'rf.pkl'), mode='wb') as f:
        pickle.dump(rf[0], f)
    with open(os.path.join(out_fp, 'nrf.pkl'), mode='wb') as f:
        pickle.dump(rf[1], f)

    print('\n-----------------------------------------------')
    print(f'Segmentation successfully trained ...')
    print('-----------------------------------------------\n')

    from cebra_em_core.segmentation.multicut import predict_mc_wf_classifiers
    res_dict = predict_mc_wf_classifiers(
        input_vols=[input_data, mem, sv, np.zeros(input_data.shape, dtype='uint8')],
        rf_filepath=os.path.join(out_fp, 'rf.pkl'),
        nrf_filepath=os.path.join(out_fp, 'nrf.pkl')
    )

    print(f'res_dict.keys() = {res_dict.keys()}')
    print(f'result.shape = {res_dict["watershed"].shape}')

    print('\n-----------------------------------------------')
    print(f'Segmentation successfully predicted ...')
    print('-----------------------------------------------\n')

    mem_filepath = os.path.join(out_fp, 'mem.h5')
    sv_filepath = os.path.join(out_fp, 'sv.h5')
    if os.path.exists(mem_filepath):
        os.remove(mem_filepath)
    with File(mem_filepath, mode='w') as f:
        f.create_dataset('data', data=mem, compression='gzip')
    if os.path.exists(sv_filepath):
        os.remove(sv_filepath)
    with File(sv_filepath, mode='w') as f:
        f.create_dataset('data', data=sv.astype('uint16'), compression='gzip')

    print('\n===============================================')
    print('All tests successful!')
    print('===============================================\n')


if __name__ == '__main__':
    test_installation()
