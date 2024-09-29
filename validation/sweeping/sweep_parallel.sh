set -x
user="???"
max_jobs=6
log_path="/path/to/honest_llama/validation/sweeping/logs"
model_name=""
model_prefix=""
instruction_prompt="default"
judge_name="???"
info_name="???"

cd /path/to/honest_llama/validation/sweeping

model_names=("llama2_chat_7B" "llama2_chat_13B" "llama3_8B_instruct")
seeds=(1 2 3)
for model_name in "${model_names[@]}"; do
    for pair in "0 1" "15 48"; do
        alpha=$(echo $pair | cut -d' ' -f1)
        K=$(echo $pair | cut -d' ' -f2)
        for seed in "${seeds[@]}"; do
            while [ $(squeue -u $USER | awk '$5 ~ /^(R|PD)$/' | wc -l) -ge $max_jobs ]; do
                echo "Maximum number of jobs ($max_jobs) reached. Waiting..."
                sleep 60
            done
            sbatch --job-name=validate_2fold_${model_prefix}${model_name}_a${alpha}_k${K}_seed${seed} \
                --output=${log_path}/${model_prefix}${model_name}_a${alpha}_k${K}_seed${seed}.out \
                --error=${log_path}/${model_prefix}${model_name}_a${alpha}_k${K}_seed${seed}.err \
                run_job_validate_2fold.sh --model_name $model_name --model_prefix "" --num_heads $K --alpha $alpha --instruction_prompt $instruction_prompt --judge_name $judge_name --info_name $info_name --seed $seed
            sleep 60
        done
    done
done