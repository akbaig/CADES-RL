#!/bin/bash
#SBATCH --job-name=cades_drl_job_cpu
#SBATCH --account=project_2008863
#SBATCH --time=08:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --partition=small

python3 -u train.py