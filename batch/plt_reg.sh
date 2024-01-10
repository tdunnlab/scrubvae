#!/usr/bin/bash
#SBATCH --job-name=plt_regs
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --mem=300G
#SBATCH --constraint=a100,sxm4  # if you want a particular type of GPU
#SBATCH --time=6-23:59
#SBATCH --output=../../ceph/results/vae/gr_parts1/plt_regs.out

source activate capture
python -u ../plot_all_reg.py