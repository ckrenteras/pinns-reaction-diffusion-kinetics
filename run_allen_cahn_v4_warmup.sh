#!/bin/bash
#SBATCH --job-name=allen_cahn_v4_warmup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/allen_cahn_v4_warmup.out
#SBATCH --error=logs/allen_cahn_v4_warmup.err

source ~/env/crunch_env/bin/activate

cd /users/ckrenter/Desktop/CRUNCH-summer-2026/pinns-reaction-diffusion-kinetics
mkdir -p logs
echo "Running allen_cahn_v4_warmup (Adam-only 1k epochs, no interp, no patience, data-only warmup for first half)"
python -u allen_cahn_v4_warmup.py
