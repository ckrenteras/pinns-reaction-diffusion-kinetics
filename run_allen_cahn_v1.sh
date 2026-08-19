#!/bin/bash
#SBATCH --job-name=allen_cahn_v1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=logs/allen_cahn_v1_%j.out
#SBATCH --error=logs/allen_cahn_v1_%j.err

source ~/env/crunch_env/bin/activate

cd /users/ckrenter/Desktop/CRUNCH-summer-2026/pinns-reaction-diffusion-kinetics
mkdir -p logs
echo "Running allen_cahn_v1"
python -u allen_cahn_v1.py