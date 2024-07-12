#!/bin/bash
#SBATCH --job-name=validate_2fold
#SBATCH --account=kempner_undergrads
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-03:00:00
#SBATCH --mem=100GB
#SBATCH --partition=kempner
#SBATCH --array=1-1

module load python
module load gcc
module load cuda
conda activate iti_h100
cd /n/holylfs06/LABS/kempner_undergrads/Users/jujipotle/honest_llama/validation

JUDGE="ft:davinci-002:harvard-university:truthful:9fudyDuw"
INFO="ft:davinci-002:harvard-university:informational:9fcfwUfc"
K=
alpha=
MODEL=

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --num_heads) K="$2"; shift ;;
        --alpha) alpha="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
python validate_2fold.py $MODEL --device 0 --num_fold 2 --use_center_of_mass --judge_name $JUDGE --info_name $INFO --num_heads $K --alpha $alpha