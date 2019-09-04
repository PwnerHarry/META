#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --time=12:0:0

# DEFAULT VALUES
RUNTIMES="240"
STEPS="20000"
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

python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 1 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 2 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 3 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 4 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 5 * $ALPHA}"`

python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.1 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.2 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.3 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.4 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.5 * $ALPHA}"`

python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 10 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 20 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 30 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 40 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 50 * $ALPHA}"`

python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.01 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.02 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.03 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.04 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.05 * $ALPHA}"`

python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 100 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 200 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 300 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 400 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 500 * $ALPHA}"`

python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.001 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.002 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.003 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.004 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.005 * $ALPHA}"`

python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 1000 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 2000 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 3000 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 4000 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 5000 * $ALPHA}"`

python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.0001 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.0002 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.0003 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.0004 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.0005 * $ALPHA}"`

python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 10000 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 20000 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 30000 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 40000 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 50000 * $ALPHA}"`

python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.00001 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.00002 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.00003 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.00004 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.00005 * $ALPHA}"`

python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 100000 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 200000 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 300000 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 400000 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --steps $STEPS --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 500000 * $ALPHA}"`