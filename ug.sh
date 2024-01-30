#!/bin/bash
#SBATCH --job-name=cdrl_test
#SBATCH --account=project_2008863
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000

python3 -u src/train.py