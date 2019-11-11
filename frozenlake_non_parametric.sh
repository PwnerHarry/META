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
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa 1e-7
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa 1e-6
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa 1e-5
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa 1e-4
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa 1e-3
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa 1e-2
python predict_frozenlake.py --parametric_lambda 0 --off_policy 1 --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --kappa 1e-1