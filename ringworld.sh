#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=48:0:0

ALPHA="0.0001"
TARGET="0.05"
BEHAVIOR="0.15"


module load python/3.7 scipy-stack
source ~/ENV/bin/activate
python -m compileall ./

python evaluate_ringworld.py --alpha $ALPHA --kappa 0.000001 --episodes 1000000 --runtimes 40 --behavior $BEHAVIOR --target $TARGET
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.00001 --episodes 1000000 --runtimes 40 --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.0001 --episodes 1000000 --runtimes 40 --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.001 --episodes 1000000 --runtimes 40 --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.01 --episodes 1000000 --runtimes 40 --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.1 --episodes 1000000 --runtimes 40 --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 1 --episodes 1000000 --runtimes 40 --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
