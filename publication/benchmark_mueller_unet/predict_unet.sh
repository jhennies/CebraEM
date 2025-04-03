#!/bin/bash
#SBATCH -A mattei
#SBATCH -C gaming
#SBATCH -p gpu-el8
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 0-0:30:00
#SBATCH --mem=16G
#SBATCH -e predict-unet.err
#SBATCH -o predict-unet.out

module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load scikit-image/0.19.3-foss-2022a

input_filepath=$1  # of the cube to be predicted
output_dirpath=$2  # Where to write the result
model_dirpath=$3  # The model file

python /g/icem/hennies/src/github/jhennies/CebraEM-benchmark/CebraEM/publication/benchmark_mueller_unet/predict_unet.py \
  "$input_filepath" \
  $output_dirpath \
  $model_dirpath \
  -src /scratch/hennies/projects/hennies/cebra-em-publication/mueller-unet/protocol-notebooks/unet \
  -v
