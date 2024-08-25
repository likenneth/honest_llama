#!/bin/bash
#SBATCH --account=???
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=0-02:00:00
#SBATCH --mem=200GB
#SBATCH --partition=???
#SBATCH --array=1-1

module load python
module load gcc
module load cuda
eval "$(conda shell.bash hook)"
conda activate iti
cd /path/to/honest_llama/validation

model_name=""
model_prefix=""
k=0
alpha=0
instruction_prompt=""
judge_name=""
info_name=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        # Parameters are only set if provided
        --model_name) model_name="${2:-$model_name}"; shift ;;
        --model_prefix) model_prefix="${2:-$model_prefix}"; shift ;;
        --num_heads) k="${2:-$k}"; shift ;;
        --alpha) alpha="${2:-$alpha}"; shift ;;
        --instruction_prompt) instruction_prompt="${2:-$instruction_prompt}"; shift ;;
        --judge_name) judge_name="${2:-$judge_name}"; shift ;;
        --info_name) info_name="${2:-$info_name}"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
echo "model_prefix: ${model_prefix}, model_name: ${model_name}, k: ${k}, alpha: ${alpha}"

if [ -z "$model_prefix" ]; then
    python validate_2fold.py --model_name $model_name --num_heads $k --alpha $alpha --instruction_prompt $instruction_prompt --device 0 --num_fold 2 --use_center_of_mass --judge_name $judge_name --info_name $info_name
else
    python validate_2fold.py --model_name $model_name --model_prefix $model_prefix --num_heads $k --alpha $alpha --instruction_prompt $instruction_prompt --device 0 --num_fold 2 --use_center_of_mass --judge_name $judge_name --info_name $info_name
fi