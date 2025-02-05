#!/bin/bash

repo_path=$1
input_dirs=$2  # e.g. path/to/gt-hela-1
run_name=$3  # e.g. unet-00
output_dirpath=$4  # e.g. path/to/unets-hela-1

echo repo_path = $repo_path

for dirpath in $input_dirs; do
  echo ''
  echo processing $dirpath
  echo ''
  echo sbatch ${repo_path}/train_unet.sh $dirpath $(basename "$dirpath")-${run_name} ${output_dirpath}
  echo ''
  sbatch ${repo_path}/train_unet.sh $dirpath $(basename "$dirpath")-${run_name} ${output_dirpath}

done

