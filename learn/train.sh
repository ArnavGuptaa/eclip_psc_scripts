#!/bin/bash

#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 01:00:00
#SBATCH --ntasks-per-node=32
#SBATCH --no-requeue

set -x

module load anaconda3 
conda activate pytorch 

echo "dataset :" + $1

echo "train full model"

python ./train.py -ds $1

echo "train motif model"

python ./train.py -ds $1 --model motif

conda deactivate