#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=48
#SBATCH --mem=8G
#SBATCH --time=16:00:00

# DEFAULT VALUES
OFF_POLICY="1"
RUNTIMES="240"
EPISODES="100000"
ALPHA="0.05"
BETA="0"

# PARSE ARGS
while [ "$1" != "" ]; do
    case $1 in
        --off_policy )            
                                shift
                                OFF_POLICY=$1
                                ;;    
        -a | --alpha )          
                                shift
                                ALPHA=$1
                                ;;
        -b | --beta )          
                                shift
                                BETA=$1
                                ;;
        -e | --episodes )       
                                shift
                                EPISODES=$1
                                ;;
        -r | --runtimes )       
                                shift
                                RUNTIMES=$1
                                ;;           
    esac
    shift
done

echo "alpha: $ALPHA, beta: $BETA"
echo "off_policy: $OFF_POLICY"
echo "runtimes: $RUNTIMES, episodes: $EPISODES"
sleep 2

# LOAD ENVIRONMENT
module load python/3.7 scipy-stack
source ~/ENV/bin/activate

# COMPILE TO ACCELERATE
python -m compileall ./

# BASELINES AND GREEDY
python predict_frozenlake.py --off_policy $OFF_POLICY --episodes $EPISODES --runtimes $RUNTIMES --alpha $ALPHA --beta $BETA --evaluate_MTA 0

# COARSE SEARCH FOR KAPPA
python predict_frozenlake.py --alpha $ALPHA --beta $BETA --kappa `awk "BEGIN {print 0.0001 * $ALPHA}"` --episodes $EPISODES --runtimes $RUNTIMES --off_policy $OFF_POLICY --evaluate_baselines 0 --evaluate_greedy 0
python predict_frozenlake.py --alpha $ALPHA --beta $BETA --kappa `awk "BEGIN {print 0.001 * $ALPHA}"` --episodes $EPISODES --runtimes $RUNTIMES --off_policy $OFF_POLICY --evaluate_baselines 0 --evaluate_greedy 0
python predict_frozenlake.py --alpha $ALPHA --beta $BETA --kappa `awk "BEGIN {print 0.01 * $ALPHA}"` --episodes $EPISODES --runtimes $RUNTIMES --off_policy $OFF_POLICY --evaluate_baselines 0 --evaluate_greedy 0
python predict_frozenlake.py --alpha $ALPHA --beta $BETA --kappa `awk "BEGIN {print 0.1 * $ALPHA}"` --episodes $EPISODES --runtimes $RUNTIMES --off_policy $OFF_POLICY --evaluate_baselines 0 --evaluate_greedy 0
python predict_frozenlake.py --alpha $ALPHA --beta $BETA --kappa `awk "BEGIN {print 1 * $ALPHA}"` --episodes $EPISODES --runtimes $RUNTIMES --off_policy $OFF_POLICY --evaluate_baselines 0 --evaluate_greedy 0