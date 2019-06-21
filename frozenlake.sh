#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=24:00:00

module load python/3.7 scipy-stack
source ~/ENV/bin/activate

python frozenlake_MTA.py --N 4 --off_policy 1 --episodes 1000000 --runtimes 240