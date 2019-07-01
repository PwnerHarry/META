#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=12:00:00

module load python/3.7 scipy-stack
source ~/ENV/bin/activate

python evaluate_frozenlake.py --off_policy 1 --episodes 1000000 --runtimes 40 --alpha 0.1 --beta 0.1 --kappa 0.1
python evaluate_frozenlake.py --off_policy 0 --episodes 1000000 --runtimes 40 --alpha 0.1 --beta 0.1 --kappa 0.1