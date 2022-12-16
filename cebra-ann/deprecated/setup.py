import runpy
from setuptools import setup, find_packages

__version__ = "0.0.1"

requires = [
    "napari[all]"
]

# optional dependencies for setuptools
extras = {
    "hdf5": "h5py",
    "zarr": "zarr",
    "n5": "pyn5",
    "cloud": "intern"
}

# dependencies only available via conda,
# we still collect them here, because the conda recipe
# gets its requirements from setuptools.
conda_only = ["vigra", "nifty", "z5py"]

extras["conda_all"] = conda_only

# NOTE in case we want to support different conda flavors at some point, we
# can add keys to 'extras', e.g. 'conda_no_hdf5' without h5py

setup(
    name="cebra-ann",
    packages=find_packages(exclude=["test"]),
    version=__version__,
    author="Julian Hennies",
    install_requires=requires,
    extras_require=extras,
    url="",
    license="MIT"
)
