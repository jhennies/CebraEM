# CebraEM

## Usage

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
 - ```init_segmentation.py```: Initializes a segmentation map
 - ```init_gt.py```: Initialize a location for annotation of ground truth
 - ```link_gt.py```: Link a ground truth cube to a segmentation dataset
 - ```log_gt.py```: Shows the ground truth cubes that are present in the current project

To annotate ground truth cubes with CebraANN use (Also see the [CebraANN readme](../cebra-ann/README.md)):

 - ```napari -w cebra-ann```

## Example workflow

TODO: Fill this with an actual example (sample dataset) including images

Assuming a EM dataset in Big Data Viewer format at ```/data/em_dataset.xml``` 
and a corresponding mask map at ```/data/labels.xml```
The EM dataset has a resolution of 5 nm isotropic.

### Initialization of the CebraEM project

```
init_project.py /data/em_dataset.xml -m /data/labels.xml
```
To change the annotation resolution, change the resolution in the general parameter settings to 10 nm. 
Press ENTER to continue.

### Run membrane prediction and supervoxels

```
run.py -t supervoxels
```

### Initialize a segmentation

Segmentations are initialized with a name (the name of the organelle, assuming mitochondria in this example) and a suffix which can be the current iteration:

```
init_segmentation.py mito it00
```

### Initialize ground truth cubes
Add ground truth cubes for the segmentation which contain the target organelle.
Look for suitable locations using the MoBIE viewer and note down the coordinates. Now run:

```
init_gt.py -b "x.xxx, y.yyy, z.zzz"
```

for each of the ground truth cubes. 

To see which ground truth cubes are already initialized run

```
log_gt.py
```

Note that all ground truth cubes have status _pending_ at this stage.

### Annotation of ground truth cubes

Before annotation the raw data, membrane prediciton and supervoxel data has to be exported. For this run

```
run.py -t gt_cubes
```

which triggers export of all initialized ground truth cubes with status _pending_.

Now run CebraANN to annotate the ground truth data (see CebraANN readme).

### Link ground truth cubes to the segmentation

Link the mitochondria segmentation of ground truth cubes 0 and 1 to the previously initialized dataset mito_it00:

```
link_gt.py 0 1 mito mito_it00
```

To show which ground truth cubes are assigned to existing segmentations run 

```
log_gt.py -d 
```

### Run the segmentation

```
run.py -t mito_it00
```

Optionally, the segmentation can be run on a subset to test the segmentation quality using the ```--roi``` parameter.

### Iterate the process

Add ground truth cubes (see Initialize ground truth cubes) specifically where the current segmentation performs poorly
and annotate using CebraANN. For this example we assume additional two cubes.

Add the segmentation for the next iteration

```
init_segmentation.py mito it01
```

Link the ground truth cubes to this second segmentation. Include the ground truth cubes of the previous iteration!

```
link_gt.py 0 1 2 3 mito mito_it01
```

Run the second iteration

```
run.py -t mito_it01
```

Then repeat with further iterations until the result is satisfactory (usually 2 to 3 iterations yield good results).

## Visualize with MoBIE

A CebraEM project is wrapped around a MoBIE project and can be directly opened within the MoBIE viewer (https://github.com/mobie/mobie-viewer-fiji) (1). 

Note that we are currently supporting MoBIE version 2.0.0 (Attached to the latest release).


## References

 1. Pape, C., Meechan, K., Moreva, E. et al. MoBIE: a Fiji plugin for sharing and exploration of multi-modal cloud-hosted big image data. Nat Methods (2023). https://doi.org/10.1038/s41592-023-01776-4
