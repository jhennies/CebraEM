
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


