#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=48:0:0

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
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 1 * $ALPHA}"`
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 2 * $ALPHA}"`
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 3 * $ALPHA}"`
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 4 * $ALPHA}"`
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 5 * $ALPHA}"`

python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.1 * $ALPHA}"`
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.2 * $ALPHA}"`
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.3 * $ALPHA}"`
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.4 * $ALPHA}"`
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.5 * $ALPHA}"`

python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.01 * $ALPHA}"`
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.02 * $ALPHA}"`
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.03 * $ALPHA}"`
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.04 * $ALPHA}"`    
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.05 * $ALPHA}"`