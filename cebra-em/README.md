# CebraEM

## Installation

For Windows use an anaconda prompt, for Linux use a terminal.

Clone the CebraEM repo into a path of your choice (now referred to as ```/path/to/cebra_em/```:

```
cd /path/of/cebra_em/
git clone https://github.com/jhennies/CebraEM.git
```

I generally recommend using the mamba package for installing the conda environments. Install mamba into the base 
environment with:

```
conda install -c conda-forge mamba
```

For the following descriptions an installation of Mamba is assumed

### Using the installation script (Linux)

Run the CebraEM installation script located in the base folder of the CebraEM repository:

```
/path/of/cebra_em/CebraEM/install.sh -m all
```

The ```-m``` flag triggers the use of mamba instead of conda.

```all``` triggers the full installation including CebraANN

On a local system where only CebraANN is needed, install with:

```
/path/of/cebra_em/CebraEM/install.sh -m ann -n cebra-ann-env
```

### Manually 

```
# Move to the CebraEM repo folder
cd /path/of/cebra_em/CebraEM

# Always create an environment with python, elf and vigra
mamba create -y -n cebra-em-env -c conda-forge python=3.9 python-elf pybdv mobie_utils=0.3 vigra bioimageio.core || exit 1
conda activate cebra-em-env

# And always install the cebra-em-core package
pip install -e ./cebra-em-core/

# Install the cebra-ann and cebra-em packages as required
pip install -e ./cebra-ann/
pip install -e ./cebra-em/

# Numba makes problems, so upgrade to version 0.54
mamba install -c conda-forge numba=0.54

# Install torch
install_torch.py
```

### Usage

Activate the CebraEM environment:

```
conda activate cebra-em-env
```

Now the CebraEM commands are available, where these can be run from any location in the file system:

 - ```convert_to_bdv.py```: Pre processing to convert a folder with tif slices, h5 volume of n5 volume to the 
    Big Data Viewer format
 - ```init_project.py```: Initializes the CebraEM project

and these are run from within a project folder:

 - ```run.py```: Computes maps or extracts ground truth cubes
 - ```init_gt.py```: Initialize a location for annotation of ground truth
 - ```link_gt.py```: Link a ground truth cube to a segmentation dataset
 - ```log_gt.py```: Shows the ground truth cubes that are present in the current project

To annotate ground truth cubes with CebraANN use (Also see CebraANN readme):

 - ```napari -w cebra-ann```



