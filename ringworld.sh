#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=24:0:0

module load python/3.7 scipy-stack
source ~/ENV/bin/activate
python evaluate_ringworld_baselines.py --episodes 1000000 --runtimes 240 --behavior 0.5 --target 0.4