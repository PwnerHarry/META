#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=48:00:00

ALPHA="0.00001"
BETA="0.000001"

module load python/3.7 scipy-stack
source ~/ENV/bin/activate

python evaluate_frozenlake.py --off_policy 1 --episodes 1000000 --runtimes 40 --alpha $ALPHA --beta $BETA --kappa 0.01 --evaluate_baselines 1 --evaluate_greedy 1
python evaluate_frozenlake.py --off_policy 0 --episodes 1000000 --runtimes 40 --alpha $ALPHA --beta $BETA --kappa 0.01 --evaluate_baselines 1 --evaluate_greedy 1

python evaluate_frozenlake.py --off_policy 1 --episodes 1000000 --runtimes 40 --alpha $ALPHA --beta $BETA --kappa 0.1 --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_frozenlake.py --off_policy 0 --episodes 1000000 --runtimes 40 --alpha $ALPHA --beta $BETA --kappa 0.1  --evaluate_baselines 0 --evaluate_greedy 0

python evaluate_frozenlake.py --off_policy 1 --episodes 1000000 --runtimes 40 --alpha $ALPHA --beta $BETA --kappa 0.001 --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_frozenlake.py --off_policy 0 --episodes 1000000 --runtimes 40 --alpha $ALPHA --beta $BETA --kappa 0.001 --evaluate_baselines 0 --evaluate_greedy 0

python evaluate_frozenlake.py --off_policy 1 --episodes 1000000 --runtimes 40 --alpha $ALPHA --beta $BETA --kappa 0.0001 --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_frozenlake.py --off_policy 0 --episodes 1000000 --runtimes 40 --alpha $ALPHA --beta $BETA --kappa 0.0001 --evaluate_baselines 0 --evaluate_greedy 0