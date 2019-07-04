#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=48:0:0

ALPHA="0.000001"

module load python/3.7 scipy-stack
source ~/ENV/bin/activate
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.0001 --episodes 1000000 --runtimes 40 --behavior 0.05 --target 0.05 --evaluate_baselines 1 --evaluate_greedy 1
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.0001 --episodes 1000000 --runtimes 40 --behavior 0.15 --target 0.05 --evaluate_baselines 1 --evaluate_greedy 1
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.0001 --episodes 1000000 --runtimes 40 --behavior 0.25 --target 0.25 --evaluate_baselines 1 --evaluate_greedy 1
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.0001 --episodes 1000000 --runtimes 40 --behavior 0.33 --target 0.25 --evaluate_baselines 1 --evaluate_greedy 1

python evaluate_ringworld.py --alpha $ALPHA --kappa 0.001 --episodes 1000000 --runtimes 40 --behavior 0.05 --target 0.05 --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.001 --episodes 1000000 --runtimes 40 --behavior 0.15 --target 0.05 --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.001 --episodes 1000000 --runtimes 40 --behavior 0.25 --target 0.25 --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.001 --episodes 1000000 --runtimes 40 --behavior 0.33 --target 0.25 --evaluate_baselines 0 --evaluate_greedy 0

python evaluate_ringworld.py --alpha $ALPHA --kappa 0.01 --episodes 1000000 --runtimes 40 --behavior 0.05 --target 0.05 --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.01 --episodes 1000000 --runtimes 40 --behavior 0.15 --target 0.05 --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.01 --episodes 1000000 --runtimes 40 --behavior 0.25 --target 0.25 --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.01 --episodes 1000000 --runtimes 40 --behavior 0.33 --target 0.25 --evaluate_baselines 0 --evaluate_greedy 0