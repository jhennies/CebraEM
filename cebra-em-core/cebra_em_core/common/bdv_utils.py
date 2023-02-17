
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
