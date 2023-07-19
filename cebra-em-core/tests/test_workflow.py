
import unittest
import numpy as np
import os
import warnings

path = os.path.dirname(__file__)


class TestWorkflow(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_pre_processing(self):
        # TODO
        assert True, 'Pre-processing does not produce the correct result'

    def test_membrane_prediction(self):

        print('\nTesting membrane prediction ...')

        input_data = np.load(os.path.join(path, 'test_data/raw00.npy'))

        from cebra_em_core.bioimageio.cebra_net import run_cebra_net
        result = run_cebra_net(input_data)

        ref = np.load(os.path.join(path, 'test_data/mem00.npy'))
        assert (result == ref).all(), 'Membrane prediction does not produce correct result'

    def test_supervoxels(self):

        print('\nTesting supervoxels ...')

        input_data = np.load(os.path.join(path, 'test_data/mem00.npy'))

        from cebra_em_core.segmentation.supervoxels import watershed_dt_with_probs
        result = watershed_dt_with_probs(input_data)

        ref = np.load(os.path.join(path, 'test_data/sv00.npy'))
        assert (result == ref).all(), 'Supervoxel computation does not produce the correct result'


if __name__ == '__main__':

    unittest.main()
