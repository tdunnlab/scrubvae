#!/usr/bin/bash
#SBATCH --job-name=spose_6fs5
#SBATCH -p mem
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --mem=3000G
#SBATCH --time=6-23:59
#SBATCH --output=./save_pose_6s5.out

source activate vae
python -u pd_fig.py #save_pose_reps.py