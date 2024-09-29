#!/bin/bash
#SBATCH --job-name=get_activations
#SBATCH --account=???
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
#SBATCH --mem=100GB
#SBATCH --partition==???

module load python
module load gcc
module load cuda
eval "$(conda shell.bash hook)"
conda activate iti
cd /path/to/honest_llama/get_activations

# Parse command-line arguments
model_name=
dataset_name=

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) model_name="$2"; shift ;;
        --dataset_name) dataset_name="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Run the activation script
python get_activations.py --model_name $model_name --dataset_name $dataset_name