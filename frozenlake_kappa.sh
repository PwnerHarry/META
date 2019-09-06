#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --time=12:0:0

# DEFAULT VALUES
RUNTIMES="240"
STEPS="1000000"
ALPHA="0.01"

# PARSE ARGS
while [ "$1" != "" ]; do
    case $1 in
        -a | --alpha )          
                                shift
                                ALPHA=$1
                                ;;
        --steps )       
                                shift
                                STEPS=$1
                                ;;
        --environment )       
                                shift
                                ENV=$1
                                ;;
        -r | --runtimes )       
                                shift
                                RUNTIMES=$1
                                ;;            
    esac
    shift
done

echo "alpha: $ALPHA"
echo "runtimes: $RUNTIMES, steps: $STEPS"
sleep 1

# LOAD ENVIRONMENT
module load python/3.7 scipy-stack
source ~/ENV/bin/activate

# COARSE SEARCH FOR KAPPA
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.005 * $ALPHA}"`
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.004 * $ALPHA}"`    
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.003 * $ALPHA}"`
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.002 * $ALPHA}"`
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.001 * $ALPHA}"`

python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.09 * $ALPHA}"`
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.08 * $ALPHA}"`    
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.07 * $ALPHA}"`
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.06 * $ALPHA}"`
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.9 * $ALPHA}"`
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.8 * $ALPHA}"`
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.7 * $ALPHA}"`
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.6 * $ALPHA}"`
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 9 * $ALPHA}"`
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 8 * $ALPHA}"`
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 7 * $ALPHA}"`
python predict_frozenlake.py --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 6 * $ALPHA}"`


