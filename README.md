# Learning Allen-Cahn-Style Reaction Diffusion Kinetics using PINNs

Hello! This is my initial attempt at using PINNs to learn reaction diffusion kinetics using physics informed neural networks, building off of Ihuaenyo et al. 2026.. This file contains my most up-to-date implementation ()denoted v4, including training script, plotting script, figures and evaluation metrics. In addition, I've included my earlier iterations. More in-depth description of the problem and my approach are detailed in the Overleaf doc.

### How to Run
1. There are batch scripts for each training file denoted: "run_allen_cahn_v{version number}.sh. These are run with the command "sbatch run_allen_cahn_v{version number}.sh"
2. PLotting is done through a separate script, which is denoted plot_v{version number}.py. This can be run directly with "python plot_v{version number}.py." It is not computation heavy, so it can be run on the login node and withou ta GPU comfortably. Results and plots are saved to results/v{version number}/

### Installation

Requires Python 3.11.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pulls the CPU-only build of `torch`. For the CUDA 12.6 build used
in development, install torch separately from the PyTorch index instead:

```bash
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```
