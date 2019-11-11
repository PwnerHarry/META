#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --time=12:0:0

# DEFAULT VALUES
RUNTIMES="240"
STEPS="50000"
ETA="1"
ALPHA="0.1"

# PARSE ARGS
while [ "$1" != "" ]; do
    case $1 in
        -a | --alpha )          
                                shift
                                ALPHA=$1
                                ;;
        --eta )          
                                shift
                                ETA=$1
                                ;;
        --steps )       
                                shift
                                STEPS=$1
                                ;;
        -r | --runtimes )       
                                shift
                                RUNTIMES=$1
                                ;;            
    esac
    shift
done

echo "runtimes: $RUNTIMES, steps: $STEPS"

module load python/3.7 scipy-stack
source ~/ENV/bin/activate

python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_MTA 0 --alpha $ALPHA --eta $ETA

python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa 1e-7
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa 1e-6
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa 1e-5
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa 1e-4
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa 1e-3
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa 1e-2
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa 1e-1