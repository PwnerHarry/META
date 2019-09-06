#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --time=12:0:0

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

# COARSE SEARCH FOR KAPPA
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.009 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.008 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.007 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.006 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0

python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.09 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.08 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.07 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.06 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0

python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.9 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.8 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.7 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 0.6 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0

python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 9 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 8 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 7 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0
python predict_ringworld.py --alpha $ALPHA --kappa `awk "BEGIN {print 6 * $ALPHA}"` --steps $STEPS --runtimes $RUNTIMES --behavior $BEHAVIOR --target $TARGET --evaluate_others 0



