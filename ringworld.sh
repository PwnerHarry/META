#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=12:00:00

module load python/3.7 scipy-stack
source ~/ENV/bin/activate
cd MTA
echo "python ringworld_MTA.py --N 11 --behavior 0.05 --target 0.05 --episodes 1000 --runtimes 160"
python ringworld_MTA.py --N 11 --behavior 0.05 --target 0.05 --episodes 1000 --runtimes 160
echo "python ringworld_MTA.py --N 11 --behavior 0.15 --target 0.05 --episodes 1000 --runtimes 160"
python ringworld_MTA.py --N 11 --behavior 0.15 --target 0.05 --episodes 1000 --runtimes 160
echo "python ringworld_MTA.py --N 11 --behavior 0.4 --target 0.4 --episodes 1000 --runtimes 160"
python ringworld_MTA.py --N 11 --behavior 0.4 --target 0.4 --episodes 1000 --runtimes 160
echo "python ringworld_MTA.py --N 11 --behavior 0.5 --target 0.4 --episodes 1000 --runtimes 160"
python ringworld_MTA.py --N 11 --behavior 0.5 --target 0.4 --episodes 1000 --runtimes 160
echo "python ringworld_MTA.py --N 11 --behavior 0.25 --target 0.25 --episodes 1000 --runtimes 160"
python ringworld_MTA.py --N 11 --behavior 0.25 --target 0.25 --episodes 1000 --runtimes 160
echo "python ringworld_MTA.py --N 11 --behavior 0.33 --target 0.25 --episodes 1000 --runtimes 160"
python ringworld_MTA.py --N 11 --behavior 0.33 --target 0.25 --episodes 1000 --runtimes 160

echo "python ringworld_MTA.py --N 25 --behavior 0.05 --target 0.05 --episodes 10000 --runtimes 80"
python ringworld_MTA.py --N 25 --behavior 0.05 --target 0.05 --episodes 10000 --runtimes 80
echo "python ringworld_MTA.py --N 25 --behavior 0.15 --target 0.05 --episodes 10000 --runtimes 80"
python ringworld_MTA.py --N 25 --behavior 0.15 --target 0.05 --episodes 10000 --runtimes 80
echo "python ringworld_MTA.py --N 25 --behavior 0.4 --target 0.4 --episodes 10000 --runtimes 80"
python ringworld_MTA.py --N 25 --behavior 0.4 --target 0.4 --episodes 10000 --runtimes 80
echo "python ringworld_MTA.py --N 25 --behavior 0.5 --target 0.4 --episodes 10000 --runtimes 80"
python ringworld_MTA.py --N 25 --behavior 0.5 --target 0.4 --episodes 10000 --runtimes 80
echo "python ringworld_MTA.py --N 25 --behavior 0.25 --target 0.25 --episodes 10000 --runtimes 80"
python ringworld_MTA.py --N 25 --behavior 0.25 --target 0.25 --episodes 10000 --runtimes 80
echo "python ringworld_MTA.py --N 25 --behavior 0.33 --target 0.25 --episodes 10000 --runtimes 80"
python ringworld_MTA.py --N 25 --behavior 0.33 --target 0.25 --episodes 10000 --runtimes 80