#!/usr/bin/bash
#SBATCH --job-name=part1
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --mem=300G
#SBATCH --constraint=a100,sxm4  # if you want a particular type of GPU
#SBATCH --time=6-23:59
#SBATCH --output=../../ceph/results/vae/gr_parts1/partspd10_rc_w51_b1_midfwd_full/train_%j_%A.out

source activate vae
python -u ../train.py "partspd10_rc_w51_b1_midfwd_full" $SLURM_ARRAY_JOB_ID $SBATCH_OUTPUT