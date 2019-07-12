#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=24:0:0

ALPHA="1"
TARGET="0.05"
BEHAVIOR="0.1"
RUNTIMES="240"
EPISODES="100000"

module load python/3.7 scipy-stack
source ~/ENV/bin/activate
python -m compileall ./

python evaluate_ringworld.py --alpha $ALPHA --kappa 0 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_MTA 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.0001 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.001 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.01 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.1 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0