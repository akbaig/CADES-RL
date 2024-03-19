#!/bin/bash
#SBATCH --job-name=cades_drl_job_gpu
#SBATCH --account=project_2008863
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --gres=gpu:v100:1

python3 -u train.py