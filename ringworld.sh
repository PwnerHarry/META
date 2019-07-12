#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=24:0:0

# DEFAULT VALUES
TARGET="0.35"
BEHAVIOR="0.4"
RUNTIMES="240"
EPISODES="100000"
ALPHA="0.05"

# PARSE ARGS
while [ "$1" != "" ]; do
    case $1 in
        -a | --alpha )          
                                shift
                                ALPHA=$1
                                ;;
        -e | --episodes )       
                                shift
                                EPISODES=$1
                                ;;
        -r | --runtimes )       
                                shift
                                RUNTIMES=$1
                                ;;
        --behavior )            
                                shift
                                BEHAVIOR=$1
                                ;;
        --target )              
                                shift
                                TARGET=$1
                                ;;                   
    esac
    shift
done

echo "alpha: $ALPHA"
echo "target: $TARGET, behavior: $BEHAVIOR"
echo "runtimes: $RUNTIMES, episodes: $EPISODES"
sleep 2

# LOAD ENVIRONMENT
module load python/3.7 scipy-stack
source ~/ENV/bin/activate

# COMPILE TO ACCELERATE
python -m compileall ./

# BASELINES AND GREEDY
python evaluate_ringworld.py --alpha $ALPHA --kappa 0 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_MTA 0

# COARSE SEARCH FOR KAPPA
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.00001 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.0001 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.001 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.01 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python evaluate_ringworld.py --alpha $ALPHA --kappa 0.1 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0