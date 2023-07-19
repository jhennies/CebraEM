
import unittest
import numpy as np
import os
import warnings


path = os.path.dirname(__file__)


class testDataset(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)

    def test_xcorr_on_volume(self):

        print('\nTesting xcorr_on_volume ...')

        from cebra_em_core.dataset.alignment import xcorr_on_volume

        input_data = np.load(os.path.join(path, 'test_data/raw00.npy'))
        result = xcorr_on_volume(input_data, median_radius=3)

        ref = np.load(os.path.join(path, 'test_data/xcorr_on_volume00.npy'))
        assert (result == ref).all(), 'Cross-correlation on volume does not produce correct result!'


if __name__ == '__main__':

    unittest.main()
