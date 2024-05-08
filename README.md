# CebraEM

## Installation

### Installation of Miniconda

Install [miniconda](https://docs.anaconda.com/free/miniconda/) on your computer.

For Windows use the miniconda prompt (open the ***"Anaconda (Miniconda3)"***-app), for Linux use a terminal.


I generally recommend using the mamba package for installing the conda environments. Install mamba into the base 
environment with:

```
conda install -c conda-forge mamba
```

For the following descriptions an installation of Mamba is assumed (otherwise just replace "mamba" -> "conda")

### Download of the CebraEM source code

Here, you have two options: 

#### 1. Download the source code in the [latest release](https://github.com/jhennies/CebraEM/archive/refs/tags/v0.0.1.zip) (recommended)

 - Now unpack the contents to some path of your choice (here I'm now assuming you created a folder called ```src``` in the user directory)
 - The path to CebraEM will now look something like this: ```~/src/CebraEM-v0.0.1/``` (Linux) or ```C:\Users\username\CebraEM-v0.0.1\``` (Windows)
 - Remember this path, this will be referred to below as ```/path/to/CebraEM```

***OR***

#### 2. Clone the CebraEM repo into a path of your choice:

```
cd /path/to/  # !!Replace the folder name!!
git clone https://github.com/jhennies/CebraEM.git
```

The path to CebraEM will be referred to below as ```/path/to/CebraEM```

### Installation of the MoBIE browser

 - Download [Fiji](https://imagej.net/software/fiji/downloads) for your operating system
 - Unpack Fiji to any location on the filesystem
 - Download the "mobie-*.jar" file from the [latest release](https://github.com/jhennies/CebraEM/releases/download/v0.0.1/mobie-2.0.0-SNAPSHOT.jar)
 - move the "mobie-*.jar" file to the ```jars``` folder in the Fiji directory (```fiji-linux64/Fiji.app/jars```)
 - Now you can open a CebraEM/MoBIE project using ```Fiji -> Plugins -> MoBIE -> Open -> Project -> Open MoBIE Project ...```

**Note: CebraEM projects currently do not work with the latest MoBIE version, so please adhere to the instructions above!**

### CebraEM conda environment

These commands install CebraEM as well as all dependencies except pytorch (run each of these lines in your terminal/prompt):
```
cd /path/to/CebraEM
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

**Now it is important to test the installation**

The following script runs the main steps of the CebraEM pipeline using a small test dataset.

```
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
