#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --time=24:0:0

# DEFAULT VALUES
TARGET="0.35"
BEHAVIOR="0.4"
RUNTIMES="240"
EPISODES="100000"
ALPHA="0.0001"

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

# LOAD ENVIRONMENT
module load python/3.7 scipy-stack
source ~/ENV/bin/activate

# COMPILE TO ACCELERATE
python -m compileall ./

python predict_ringworld.py --alpha $ALPHA --kappa 6.1e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 6.2e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 6.3e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 6.4e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 6.5e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 6.6e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 6.7e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 6.8e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 6.9e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 7.0e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 7.1e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 7.2e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 7.3e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 7.4e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 7.5e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 7.6e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 7.7e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 7.8e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
python predict_ringworld.py --alpha $ALPHA --kappa 7.9e-5 --episodes $EPISODES --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_baselines 0 --evaluate_greedy 0
