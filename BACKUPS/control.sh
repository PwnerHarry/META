#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=8G
#SBATCH --time=12:0:0

# DEFAULT VALUES
ENV="cartpole"
RUNTIMES="240"
EPISODES="1000"
ALPHA="0.01"

# PARSE ARGS
while [ "$1" != "" ]; do
    case $1 in
        --episodes )       
                                shift
                                EPISODES=$1
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

echo "env: $ENV, runtimes: $RUNTIMES, episodes: $EPISODES"

for ALPHA in 0.01 0.02 0.03 0.04 0.05 0.001 0.002 0.003 0.004 0.005 0.0001 0.0002 0.0003 0.0004 0.0005
do
    python control_$ENV.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_MTA 0 --alpha $ALPHA

    python control_$ENV.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 1 * $ALPHA}"`
    python control_$ENV.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 2 * $ALPHA}"`
    python control_$ENV.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 3 * $ALPHA}"`
    python control_$ENV.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 4 * $ALPHA}"`
    python control_$ENV.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 5 * $ALPHA}"`

    python control_$ENV.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.1 * $ALPHA}"`
    python control_$ENV.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.2 * $ALPHA}"`
    python control_$ENV.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.3 * $ALPHA}"`
    python control_$ENV.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.4 * $ALPHA}"`
    python control_$ENV.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.5 * $ALPHA}"`

    python control_$ENV.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.01 * $ALPHA}"`
    python control_$ENV.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.02 * $ALPHA}"`
    python control_$ENV.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.03 * $ALPHA}"`
    python control_$ENV.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.04 * $ALPHA}"`
    python control_$ENV.py --runtimes $RUNTIMES --episodes $EPISODES --evaluate_baselines 0 --evaluate_greedy 0 --alpha $ALPHA --kappa `awk "BEGIN {print 0.05 * $ALPHA}"`
done