#!/bin/bash
#SBATCH -A mattei
#SBATCH -C gaming
#SBATCH -p gpu-el8
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 1-0:00:00
#SBATCH --mem=16G
#SBATCH -e train-unet.%N.%j.err
#SBATCH -o train-unet.%N.%j.out

module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load scikit-image/0.19.3-foss-2022a

train_data_dirpath=$1
unet_name=$2
out_dirpath=$3

python /g/icem/hennies/src/github/jhennies/CebraEM-benchmark/CebraEM/publication/benchmark_mueller_unet/train_unet.py \
  $train_data_dirpath \
  -name $unet_name \
  -src /scratch/hennies/projects/hennies/cebra-em-publication/mueller-unet/protocol-notebooks/unet \
  -out $out_dirpath \
  -v
