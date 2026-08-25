#!/bin/bash
#SBATCH --job-name=allen_cahn_v4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/allen_cahn_v4.out
#SBATCH --error=logs/allen_cahn_v4.err

source ~/env/crunch_env/bin/activate

cd /users/ckrenter/Desktop/CRUNCH-summer-2026/pinns-reaction-diffusion-kinetics
mkdir -p logs
echo "Running allen_cahn_v4"
python -u allen_cahn_v4.py
