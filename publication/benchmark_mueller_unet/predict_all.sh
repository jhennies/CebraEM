#!/bin/bash

"""
sbatch /g/icem/hennies/src/github/jhennies/CebraEM-benchmark/CebraEM/publication/benchmark_mueller_unet/predict_unet.sh \
    val-hela-1-bin2/hela-1-10nm-011-val-raw.tif \
    hela-1-predict-mito-bin2-00 \
    unets-hela-1-bin2/models/mito-unet-00/
"""


repo_path=$1
input_files=$2  # e.g. path/to/gt-hela-1
run_name=$3  # e.g. hela-1-predict-mito-bin2-00
model_dirpath=$4  # e.g. unets-hela-1-bin2/models/mito-unet-00/

echo repo_path = $repo_path

for filepath in $input_files; do
  echo ''
  echo processing $filepath
  echo ''
  echo sbatch /g/icem/hennies/src/github/jhennies/CebraEM-benchmark/CebraEM/publication/benchmark_mueller_unet/predict_unet.sh
  echo "  ${filepath}"
  echo "  $(pwd)/${run_name}"
  echo "  ${model_dirpath}"

  sbatch /g/icem/hennies/src/github/jhennies/CebraEM-benchmark/CebraEM/publication/benchmark_mueller_unet/predict_unet.sh \
    ${filepath} \
    $(pwd)/${run_name} \
    ${model_dirpath}

done

#
#for dirpath in $input_dirs; do
#  echo ''
#  echo processing $dirpath
#  echo ''
#  echo sbatch ${repo_path}/train_unet.sh $dirpath $(basename "$dirpath")-${run_name} ${output_dirpath}
#  echo ''
#  sbatch ${repo_path}/train_unet.sh $dirpath $(basename "$dirpath")-${run_name} ${output_dirpath}
#
#done

