#!/bin/bash
#SBATCH --job-name=cdrl_test_gpu
#SBATCH --account=project_2008863
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1

python3 -u src/train.py