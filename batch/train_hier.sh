#!/usr/bin/bash
#SBATCH --job-name=hr51_o0
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --mem=120G
#SBATCH --constraint=a100,sxm4  # if you want a particular type of GPU
#SBATCH --time=6-23:59
#SBATCH --output=../results/hr_w51_b15_midfwd_full_o0/train.out

source activate vae
python -u ../train_hier.py