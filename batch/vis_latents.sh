#!/usr/bin/bash
#SBATCH --job-name=grm_aug
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --mem=500G
#SBATCH --constraint=a100,sxm4   # if you want a particular type of GPU
#SBATCH --time=6-23:59
#SBATCH --output=../../ceph/results/vae/avgspd_grm_rc_w51_midfwd_full/spd_aug.out

source activate vae
# python -u ../vis_latents.py
python -u ../test_spd_aug.py
# python -u ../vis_hier.py