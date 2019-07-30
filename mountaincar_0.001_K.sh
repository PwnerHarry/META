#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --time=24:0:0

# DEFAULT VALUES
RUNTIMES="240"
EPISODES="10000"

# PARSE ARGS
while [ "$1" != "" ]; do
    case $1 in
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

for ALPHA in 0.001 0.002 0.003 0.004 0.005
do
    # python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_MTA 0 --alpha $ALPHA

    python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 1 * $ALPHA}"`
    python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 2 * $ALPHA}"`
    python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 3 * $ALPHA}"`
    python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 4 * $ALPHA}"`
    python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 5 * $ALPHA}"`

    python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.1 * $ALPHA}"`
    python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.2 * $ALPHA}"`
    python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.3 * $ALPHA}"`
    python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.4 * $ALPHA}"`
    python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.5 * $ALPHA}"`

    python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.01 * $ALPHA}"`
    python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.02 * $ALPHA}"`
    python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.03 * $ALPHA}"`
    python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.04 * $ALPHA}"`
    python control_mountaincar.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.05 * $ALPHA}"`
done