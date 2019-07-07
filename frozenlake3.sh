#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=16:00:00

ALPHA="0.00001"
BETA="0.00001"

module load python/3.7 scipy-stack
source ~/ENV/bin/activate
python -m compileall ./

python evaluate_frozenlake.py --off_policy 0 --episodes 1000000 --runtimes 240 --alpha $ALPHA --beta $BETA --evaluate_MTA 0
python evaluate_frozenlake.py --off_policy 1 --episodes 1000000 --runtimes 240 --alpha $ALPHA --beta $BETA --evaluate_MTA 0