#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=8G
#SBATCH --time=12:0:0

python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.001 --kappa 0.002
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.001 --kappa 0.003
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.001 --kappa 0.004
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.001 --kappa 0.005

python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.001 --kappa 0.0001
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.001 --kappa 0.0002
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.001 --kappa 0.0003
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.001 --kappa 0.0004
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.001 --kappa 0.0005

python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.01 --kappa 0.04
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.01 --kappa 0.05
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.01 --kappa 0.001
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.01 --kappa 0.002
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.01 --kappa 0.003
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.01 --kappa 0.004
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.01 --kappa 0.005