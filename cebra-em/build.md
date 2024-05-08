# How to build the squirrel package

**Create and activate the conda-build environment**

```
conda create -n conda-build-env conda-build
conda activate conda-build-env
```

optional:

```
conda install -c conda-forge mamba
mamba install -c conda-forge boa
```

**Build the squirrel package**

Either:

```
cd /path/to/squirrel
conda-build .
```

or if boa was installed above:

```
cd /path/to/squirrel
conda mambabuild .
```

**Install the squirrel package to a local environment**

Use a new terminal or deactivate the conda-build environment

```
conda create -n squirrel-env
conda install -c /path/to/miniconda3/envs/conda-build-env/conda-bld/ squirrel
```

**Important:** If you install the package again after changing some code and re-building but WITHOUT changing the 
version, make sure you delete the existing squirrel package in ```/path/to/miniconda3/pkgs```.
After updating the dependencies, also delete the build cache in ```/path/to/miniconda3/conda-build-env/conda-bld/```
