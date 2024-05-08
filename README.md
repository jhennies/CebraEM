# CebraEM

## Installation

### Pre-requesites

Install [miniconda](https://docs.anaconda.com/free/miniconda/) on your computer.

For Windows use the miniconda prompt (open the ***"Anaconda (Miniconda3)"***-app), for Linux use a terminal.

Clone the CebraEM repo into a path of your choice (now referred to as ```/path/to/cebra_em/```):

```
cd /path/to/cebra_em/
git clone https://github.com/jhennies/CebraEM.git
```

I generally recommend using the mamba package for installing the conda environments. Install mamba into the base 
environment with:

```
conda install -c conda-forge mamba
```

For the following descriptions an installation of Mamba is assumed (otherwise just replace "mamba" -> "conda")

### CebraEM conda environment

These commands install CebraEM as well as all dependencies except pytorch (run each of these lines in your terminal/prompt):
```
mamba create -y -n cebra-em-env -c conda-forge python=3.9 python-elf pybdv mobie_utils=0.3 vigra bioimageio.core=0.5.11 bioimageio.spec=0.4.9.post5 marshmallow
conda activate cebra-em-env

pip install -e ./cebra-em-core/
pip install -e ./cebra-ann/
pip install -e ./cebra-em/
```

For pytorch you will need to determine your Cuda version and select the correct installation call. 
Please refer to https://pytorch.org/get-started/locally/ for further information.
As an example, for our system with Cuda version 11, we installed pytorch successfully using this pip install command: 

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


### Testing the installation

The following script runs the main steps of the CebraEM pipeline using a small test dataset.

```
conda activate cebra-em-env
python test_installation.py
```

Possible errors:

```
ModuleNotFoundError: No module named 'torch'
```

The installation of pytorch is missing or did not work properly. 
Run ```nvidia-smi``` to determine your Cuda version and refer to https://pytorch.org/get-started/locally/ to determine 
the proper pytorch installation commands

## CebraEM

[CebraEM readme](cebra-em/README.md)

Contains:
 - Usage of CebraEM

## CebraANN

[CebraANN readme](cebra-ann/README.md)

Contains:
 - Specific installation for the cebra-ann package in case CebraANN needs to be run on a separate machine
 - Usage of CebraANN

## CebraNET

[CebraNET readme](CebraNET_README.md)
