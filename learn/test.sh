#!/bin/bash

#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 01:00:00
#SBATCH --ntasks-per-node=32
#SBATCH --no-requeue

set -x

module load anaconda3 
conda activate pytorch 

echo "model :" + $1

echo "dataset :" + $2

echo "testset :" + $3

echo "test full model"

python ./test.py --model --testfasta $3 -ds $2

# echo "test motif model"

# python ./train.py -ds $1 --model motif

conda deactivate