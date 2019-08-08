#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --nodes=1
#SBATCH --time=12:0:0

# DEFAULT VALUES
RUNTIMES="240"
EPISODES="10000"
ETA="1"
ALPHA="0.1"

# PARSE ARGS
while [ "$1" != "" ]; do
    case $1 in
        -a | --alpha )          
                                shift
                                ALPHA=$1
                                ;;
        --episodes )       
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

echo "runtimes: $RUNTIMES, episodes: $EPISODES"

module load python
source ~/ENV/bin/activate

python -m compileall ./

python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_MTA 0 --alpha $ALPHA --eta $ETA

python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 1 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 2 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 3 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 4 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 5 * $ALPHA}"`

python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.1 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.2 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.3 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.4 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.5 * $ALPHA}"`

python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.01 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.02 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.03 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.04 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.05 * $ALPHA}"`

python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.001 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.002 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.003 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.004 * $ALPHA}"`
python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_others 0 --alpha $ALPHA --eta $ETA --kappa `awk "BEGIN {print 0.005 * $ALPHA}"`