#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=48:0:0

# DEFAULT VALUES
TARGET="0.35"
BEHAVIOR="0.4"
RUNTIMES="240"
STEPS="1000000"
ALPHA="0.05"

# PARSE ARGS
while [ "$1" != "" ]; do
    case $1 in
        -a | --alpha )          
                                shift
                                ALPHA=$1
                                ;;
        -e | --steps )       
                                shift
                                STEPS=$1
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
echo "runtimes: $RUNTIMES, steps: $STEPS"
sleep 1

# LOAD ENVIRONMENT
module load python/3.7 scipy-stack
source ~/ENV/bin/activate

# BASELINES AND LAMBDA_GREEDY
python predict_ringworld.py --alpha $ALPHA --kappa 0 --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_MTA 0

# COARSE SEARCH FOR KAPPA
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 1 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 2 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 3 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 4 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 5 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0

python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.1 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.2 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.3 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.4 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.5 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0

python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.01 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.02 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.03 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.04 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.05 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
