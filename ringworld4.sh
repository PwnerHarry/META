#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=16:0:0

TARGET="0.05"
BEHAVIOR="0.1"
RUNTIMES="240"
EPISODES="100000"

module load python/3.7 scipy-stack
source ~/ENV/bin/activate
python -m compileall ./

python evaluate_ringworld.py --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_MTA 0 --alpha 0.0003
python evaluate_ringworld.py --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_MTA 0 --alpha 0.0004
python evaluate_ringworld.py --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_MTA 0 --alpha 0.0005
python evaluate_ringworld.py --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_MTA 0 --alpha 0.0006
python evaluate_ringworld.py --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_MTA 0 --alpha 0.0007
python evaluate_ringworld.py --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_MTA 0 --alpha 0.0008
python evaluate_ringworld.py --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_MTA 0 --alpha 0.0009