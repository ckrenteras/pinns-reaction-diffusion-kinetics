#!/bin/bash
#SBATCH --job-name=diagnose_collapse
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/diagnose_collapse.out
#SBATCH --error=logs/diagnose_collapse.err

source ~/env/crunch_env/bin/activate

cd /users/ckrenter/Desktop/CRUNCH-summer-2026/pinns-reaction-diffusion-kinetics
mkdir -p logs
echo "Running diagnose_collapse"
python -u diagnose_collapse.py
