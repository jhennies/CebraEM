# CebraEM

## Installation

### Support for Linux and Windows

Note that CebraEM is developed on Linux. We have tested CebraEM on Windows and are currently working
on a stable solution for Windows machines. 

Note that for Windows 10, a Windows Subsystem for Linux ([WSL](https://learn.microsoft.com/en-us/windows/wsl/about))
has to be installed for CebraEM to run. 
In Windows 11 this is already installed by default.

### Installation of the MoBIE browser

Follow the instruction [here](https://github.com/mobie/mobie-viewer-fiji/) to install the Fiji and the MoBIE browser 
which includes these steps:

 - Download [Fiji](https://imagej.net/software/fiji/downloads) for your operating system
 - Unpack Fiji to any location on the filesystem
 - Open Fiji and install MoBIE:
   - Open the Fiji upater using ```Help -> Update...``` and wait for the updater to start up (may take a moment)
   - An information prompt will probably report that 'Your ImageJ is up to date!'. Press ```OK```
   - In the "ImageJ Updater" window click ```Manage Update Sites```
   - In the "Manage Update Sites" window scroll down or search for "MoBIE" and tick the box next to it
   - Press ```Apply and Close```
   - Back in the "ImageJ Updater" window click ```Apply Changes``` which will now trigger download and installation of 
     the required packages
   - Close and re-open Fiji to complete the installation
 
Alternatively, can download and extract a Fiji that already includes MoBIE from the [latest release](TODO: put link).

After successful installation of MoBIE you can open a CebraEM/MoBIE project using ```Fiji -> Plugins -> MoBIE -> Open -> Project -> Open MoBIE Project ...```

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

### Installation of Miniconda

Install [miniconda](https://docs.anaconda.com/free/miniconda/) on your computer.

I generally recommend using the mamba package for installing the conda environments. Open a terminal and install mamba into the base 
environment with:

```
conda install -c conda-forge mamba
```

For the following descriptions an installation of Mamba is assumed (otherwise just replace "mamba" -> "conda")

### CebraEM conda environment

These commands install CebraEM as well as all dependencies except pytorch (run each of these lines in your terminal/prompt):

```
cd /path/to/CebraEM
mamba env create -f environment.yaml
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
