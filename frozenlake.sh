#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=12:00:00

module load python/3.7 scipy-stack
source ~/ENV/bin/activate

python evaluate_frozenlake.py --off_policy 1 --episodes 1000000 --runtimes 40 --learner_type togtd --alpha 0.05 --beta 0.05 --kappa 0.025
python evaluate_frozenlake.py --off_policy 0 --episodes 1000000 --runtimes 40 --learner_type togtd --alpha 0.05 --beta 0.05 --kappa 0.025