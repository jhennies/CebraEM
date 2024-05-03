# CebraANN

## Installation

For a full CebraEM installation including CebraANN, please follow the installation instruction in the [main readme](../README.md). 

In case you are installing CebraANN on a machine separate from the main CebraEM workflow, 
you can omit the installation of the cebra-em package as well as pytorch. 

The simplified installation for CebraANN looks like this:

```
mamba create -y -n cebra-ann-env -c conda-forge python=3.9 python-elf pybdv mobie_utils=0.3 vigra bioimageio.core=0.5.11 bioimageio.spec=0.4.9.post5 marshmallow
conda activate cebra-ann-env

pip install -e ./cebra-em-core/
pip install -e ./cebra-ann/
```

## Usage

Start the software with

```
conda activate cebra-em-env  # or cebra-ann-env, depending on your installation
napari -w cebra-ann
```

Note: The intended usage centers around the annotation of ground truth cubes extracted by the CebraEM workflow. 
This will be the main focus of this description here. However, there is additional functionality build in that 
enables a stand-alone usage of CebraANN without the CebraEM project framework.

The general CebraANN workflow works like so:

1. **Inputs tab:** After starting the software click ```Load project``` and select a ground truth folder from the CebraEM project located in 
```proj_path/gt/```. These directories contain three files (```raw.h5```, ```mem.h5``` and ```sv.h5```) which are 
required by the default CebraANN annotation workflow.
2. **Pre-merging tab:** Select a value for ```Beta``` and click ```Compute```. The value for Beta can be between ```0``` 
(no merges) and ```1``` (all is merged) and can be selected by personal preference. The result will be manually curated 
in the following step.
3. **Instance segmentation tab:** When happy with the pre-merging step click ```Start``` to activate the instance 
segmentation workflow. An additionally layer labelled "instances" will appear. Whenever this layer is selected, the 
following functionality is available:
   - ```CTRL + LeftMouseDown + MouseMove```: Merge full instances by drawing
   - ```CTRL + SHIFT + LeftMouseDown + MouseMove```: Add single supervoxels to an instance
   - ```CTRL + RightMouseDown + MouseMove```: Create a new instance
4. **Semantic segmentation tab:** Every instance that is completed in the instance segmentation step can be moved to 
its own semantic layer. 
   1. Select an organelle (or "< other >" for a custom organelle) in the drop-down
   2. Click ```Add``` such that a new segmentation layer is added named "semantics_organelle"
   3. Select the layer that an instance should be moved to and click on this instance in the viewer. The instance is now
deleted from the instances layer and moved to its semantics layer. Moving it back (in case further modification is 
required) works accordingly.
5. When ready, click ```Save Project``` which triggers the export of the segmentations. When working within a CebraEM
project the results are saved to the location where the CebraEM scripts are expecting them to be and thus will be 
automatically recognized

Steps 3 and 4 can be performed alongside until all required instances are segmented fully and assigned to their 
respective semantic classes.
