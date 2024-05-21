#!/usr/bin/bash
#SBATCH --job-name=procrustes
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --mem=250G
#SBATCH --constraint=a100|v100   # if you want a particular type of GPU
#SBATCH --time=6-23:59
#SBATCH --output=./results/pd_fig.out

source activate vae
# python -u ./procrustes.py
python -u ./pd_fig.py
# python -u ../test_spd_aug.py
# python -u ../vis_hier.py