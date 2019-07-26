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

python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 1.1
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 1.2
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 1.3
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 1.4
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 1.5
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 1.6
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 1.7
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 1.8
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 1.9
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 2.0
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 2.1
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 2.2
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 2.3
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 2.4
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa 2.5