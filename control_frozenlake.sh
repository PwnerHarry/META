#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=8G
#SBATCH --time=12:0:0

python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.00001 --kappa 0.00001
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.00002 --kappa 0.00002
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.00003 --kappa 0.00003
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.00004 --kappa 0.00004
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.00005 --kappa 0.00005

python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.0001 --kappa 0.0001
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.0002 --kappa 0.0002
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.0003 --kappa 0.0003
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.0004 --kappa 0.0004
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.0005 --kappa 0.0005

python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.001 --kappa 0.001
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.002 --kappa 0.002
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.003 --kappa 0.003
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.004 --kappa 0.004
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.005 --kappa 0.005

python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.01 --kappa 0.01
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.02 --kappa 0.02
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.03 --kappa 0.03
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.04 --kappa 0.04
python control_frozenlake.py --runtimes 16 --evaluate_baselines 0 --evaluate_greedy 0 --episodes 100000 --alpha 0.05 --kappa 0.05