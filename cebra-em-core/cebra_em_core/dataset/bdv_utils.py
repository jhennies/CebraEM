
import os
import numpy as np
import re
import time
from glob import glob

import xml.etree.ElementTree as ET
from pybdv.metadata import get_data_path
from pybdv.util import HDF5_EXTENSIONS
from pybdv.util import get_key, open_file
from pybdv.converter import normalize_output_path
from pybdv.metadata import write_n5_metadata, write_h5_metadata, write_xml_metadata, validate_attributes
from pybdv.bdv_datasets import BdvDataset


def is_h5(xml_path):
    path = get_data_path(xml_path)
    return os.path.splitext(path)[1].lower() in HDF5_EXTENSIONS


def get_shape(xml_path, setup_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    seqdesc = root.find('SequenceDescription')
    viewsets = seqdesc.find('ViewSetups')
    vsetups = viewsets.findall('ViewSetup')
    for vs in vsetups:
        if vs.find('id').text == str(setup_id):
            shape = vs.find('size').text
            return [float(shp) for shp in shape.split()][::-1]
    raise ValueError("Could not find setup %i" % setup_id)


def create_empty_dataset(
        data_path, setup_id, timepoint,
        data_shape,
        data_dtype='uint8',
        chunks=None,
        scale_factors=None,
        resolution=(1., 1., 1.),
        unit='pixel',
        setup_name=None,
        attributes=None,
        verbose=False
):

    if verbose:
        print('data_shape = {}'.format(data_shape))
        print('resolution = {}'.format(resolution))
        print('unit = {}'.format(unit))

    data_path, xml_path, is_h5 = normalize_output_path(data_path)
    if verbose:
        print('data_path = {}'.format(data_path))
        print('xml_path = {}'.format(xml_path))
        print('is_h5 = {}'.format(is_h5))

    if attributes is None:
        attributes = {'channel': {'id': None}}

    # validate the attributes
    enforce_consistency = True
    attributes_ = validate_attributes(xml_path, attributes, setup_id, enforce_consistency)

    base_key = get_key(is_h5, timepoint=timepoint, setup_id=setup_id, scale=0)
    with open_file(data_path, 'a') as f:

        # Create the full sized dataset
        f.create_dataset(base_key, shape=data_shape, compression='gzip',
                         chunks=chunks, dtype=data_dtype)

        # Create additional scales
        factors = []
        factor = np.array([1, 1, 1])
        factors.append(factor.tolist())

        if scale_factors is not None:

            for scale_id, scale_factor in enumerate(scale_factors):

                factor *= np.array(scale_factor)
                factors.append(scale_factor)

                base_key = get_key(is_h5, timepoint=timepoint, setup_id=setup_id, scale=scale_id + 1)

                f.create_dataset(base_key, shape=(np.array(data_shape) / factor).astype(int),
                                 compression='gzip', chunks=chunks, dtype=data_dtype)

        print(f'factors = {factor}')

    if is_h5:
        write_h5_metadata(data_path, factors, setup_id, timepoint, overwrite=False)
    else:
        write_n5_metadata(data_path, factors, resolution, setup_id, timepoint, overwrite=False)

    # write bdv xml metadata
    print(f'setup_id = {setup_id}')
    print(f'setup_name = {setup_name}')
    print(f'xml_path = {xml_path}')
    write_xml_metadata(xml_path, data_path, unit,
                       resolution, is_h5,
                       setup_id=setup_id,
                       timepoint=timepoint,
                       setup_name=setup_name,
                       affine=None,
                       attributes=attributes_,
                       overwrite=False,
                       overwrite_data=False,
                       enforce_consistency=enforce_consistency)

    return False


class BdvDatasetAdvanced(BdvDataset):

    def __init__(
            self,
            path,
            timepoint,
            setup_id,
            downscale_mode='mean',
            halo=None,
            background_value=None,
            unique=False,
            update_max_id=False,
            n_threads=1,
            verbose=False
    ):

        self._halo = halo
        self._background_value = background_value
        self._unique = unique
        self._update_max_id = update_max_id

        if unique:
            self._update_max_id = True

        super().__init__(
            path,
            timepoint,
            setup_id,
            downscale_mode=downscale_mode,
            n_threads=n_threads,
            verbose=verbose
        )

    def set_halo(self, halo):
        """
        Adjust the halo any time you want
        """
        self._halo = halo

    def set_max_id(self, idx, compare_with_present=False):
        """
        Use this to update the largest present id in the dataset if you employ a stitching method with unique == True.
        The id is automatically updated if new data is written.
        """
        data_path = self._path
        with open_file(data_path, 'a') as f:
            key = get_key(self._is_h5, self._timepoint, self._setup_id, 0)
            if not compare_with_present or f[key].attrs['maxId'] < idx:
                f[key].attrs['maxId'] = idx

    def get_max_id(self):
        data_path = self._path
        with open_file(data_path, 'r') as f:
            key = get_key(self._is_h5, self._timepoint, self._setup_id, 0)
            max_id = f[key].attrs['maxId']
        return max_id

    def _crop(self, dd, volume, unique):
        """
        Writes the data to the dataset while cropping away the halo and updating the max id
        """

        if unique:
            max_id = self.get_max_id()
        else:
            max_id = 0

        position = [d.start for d in dd]
        shp = [d.stop - d.start for d in dd]
        assert list(volume.shape) == shp

        halo = self._halo

        if halo is not None:
            volume = volume[
                     halo[0]: -halo[0],
                     halo[1]: -halo[1],
                     halo[2]: -halo[2]
                     ]
            if unique:
                if self._background_value is not None:
                    assert self._background_value == 0, 'Only implemented for background value == 0'
                    volume[volume != 0] = volume[volume != 0] + max_id
                else:
                    volume += max_id + 1

            dd = np.s_[
                position[0] + halo[0]: position[0] + shp[0] - halo[0],
                position[1] + halo[1]: position[1] + shp[1] - halo[1],
                position[2] + halo[2]: position[2] + shp[2] - halo[2]
            ]

        else:
            if unique:
                if self._background_value is not None:
                    assert self._background_value == 0, 'Only implemented for background value == 0'
                    volume[volume != 0] = volume[volume != 0] + max_id
                else:
                    volume += max_id + 1

            dd = np.s_[
                 position[0]: position[0] + shp[0],
                 position[1]: position[1] + shp[1],
                 position[2]: position[2] + shp[2]
                 ]

        return dd, volume

    def __setitem__(self, key, value):

        # Get key and value to write
        key, value = self._crop(key, value, self._unique)

        # Update the maximum label
        if self._update_max_id:
            vol_max = int(value.max())
            print(f'Updating max_id to: {vol_max}')
            self.set_max_id(vol_max, compare_with_present=True)

        # Now call the super with the properly stitched volume
        super().__setitem__(key, np.array(value))


def bdv2pos(pos, resolution=None, verbose=False):
    pos = np.array([float(x) for x in (re.findall('\d+\.\d+', pos))])[::-1]
    if verbose:
        print(f'pos = {pos}')
    if resolution is not None:
        pos = np.round(pos / np.array(resolution)).astype(int)
    return pos
