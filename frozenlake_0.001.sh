#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --time=24:0:0

# DEFAULT VALUES
RUNTIMES="240"
EPISODES="100000"
ALPHA="0.001"

# LOAD ENVIRONMENT
module load python/3.7 scipy-stack
source ~/ENV/bin/activate

# COMPILE TO ACCELERATE
python -m compileall ./

python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.3
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.35
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.4
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.45
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.5
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.55
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.6
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.65
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.7
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.75
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.8
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.85
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.9
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.95
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 1

python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.015
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.016
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.014
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.0155
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.0145
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.01525
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.01475
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.015125
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 0.014875