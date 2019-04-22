#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=12:00:00

module load python/3.7 scipy-stack
source ~/ENV/bin/activate
cd MTA
echo "python frozenlake_MTA.py --N 4 --off_policy 1 --episodes 1000000 --runtimes 160"
python frozenlake_MTA.py --N 4 --off_policy 1 --episodes 10000 --runtimes 160

echo "python frozenlake_MTA.py --N 4 --off_policy 1 --episodes 1000000 --runtimes 160"
python frozenlake_MTA.py --N 4 --off_policy 0 --episodes 10000 --runtimes 160
