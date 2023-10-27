# CebraNET

Cellular membrane prediction model for volume SEM datasets. This model was trained on a FIB-SEM dataset to generically predict membranes (or organelle boundaries) in any volume SEM dataset.

The model was published as major component of the [CebraEM](https:/github.com/jhennies/CebraEM) workflow in [hennies et al. 2023](https://www.biorxiv.org/content/10.1101/2023.04.06.535829v1).

## Usage of CebraNET independent of CebraEM

CebraNET is available  in the Bioimage Model Zoo (bioimage.io)
([CebraNET @bioimage.io](https://bioimage.io/#/?id=10.5281%2Fzenodo.7274275), [CebraNET @zenodo](https://zenodo.org/record/7274276))
where it runs in: 

 - [Ilastik](https://www.ilastik.org/)
 - [Deep ImageJ (upcoming)](https://deepimagej.github.io/)

To optimize the output of CebraNET for your particular workflow, also see the following section about tweaking the input data.

## Input data and pre-processing

Dependent on your input data, you might want to consider the following points as applicable:

1. Although CebraNET is robust towards image noise sometimes a slight Gaussian smoothing 
   can reduce the amount of false positive predictions
2. CebraNET was trained to cope with mis-alignments of the input data stack. 
   However this only works to a certain extent (few pixels), so consider locally re-aligning your data stack if the 
   membrane prediction quality suffers
3. Use isotropic data! If your data is anisotropic, re-scale it. For the SBEM dataset used in 
   [hennies et al. 2023](https://www.biorxiv.org/content/10.1101/2023.04.06.535829v1), we scaled the data from 10 x 10 x 25 nm resolution to 10 nm isotropic resolution.
4. Scale your data to alter the level of detail predicted by CebraNET. 
   A resolution of 10 nm is well suited for larger organelles such as mitochondria.
   A resolution of 5 nm usually yields high level of detail to reconstruct the ER or other small, fine structured organelles.



