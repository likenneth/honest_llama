set -x
user="????"
max_jobs=6
log_path="/path/to/honest_llama/validation/sweeping/logs"
model_name="llama_7B"
model_prefix=""
instruction_prompt="default"
judge_name="???"
info_name="???"

cd /path/to/honest_llama/validation/sweeping

# Baseline
sbatch --job-name=validate_2fold_${model_prefix}${model_name}_a${1}_k${0} \
        --output=${log_path}/${model_prefix}${model_name}_a${1}_k${0}.out \
        --error=${log_path}/${model_prefix}${model_name}_a${1}_k${0}.err \
        run_job_validate_2fold.sh --model_name $model_name --model_prefix $model_prefix --num_heads 1 --alpha 0 --instruction_prompt $instruction_prompt --judge_name $judge_name --info_name $info_name
sleep 10

for alpha in 15; do
    for K in 48; do
        while [ $(squeue -u $USER | awk '$5 ~ /^(R|PD)$/' | wc -l) -ge $max_jobs ]; do
            echo "Maximum number of jobs ($max_jobs) reached. Waiting..."
            sleep 60
        done
        sbatch --job-name=validate_2fold_${model_prefix}${model_name}_a${alpha}_k${K} \
               --output=${log_path}/${model_prefix}${model_name}_a${alpha}_k${K}.out \
               --error=${log_path}/${model_prefix}${model_name}_a${alpha}_k${K}.err \
               run_job_validate_2fold.sh --model_name $model_name --model_prefix $model_prefix --num_heads $K --alpha $alpha --instruction_prompt $instruction_prompt --judge_name $judge_name --info_name $info_name
        sleep 10
    done
done


# To sweep across all models (Baseline, ITI, Baked-in ITI):
model_names=("llama_7B" "llama2_chat_7B" "llama2_chat_13B" "llama2_chat_70B" "llama3_8B_instruct" "llama3_70B_instruct")

# for model_name in "${model_names[@]}"; do
#     while [ $(squeue -u $user | awk '$5 ~ /^(R|PD)$/' | wc -l) -ge $max_jobs ]; do
#         echo "Maximum number of jobs ($max_jobs) reached. Waiting..."
#         sleep 60
#     done

#     sbatch --job-name=validate_2fold_${model_name}_a0_k1 \
#             --output=${log_path}/${model_name}_a0_k1.out \
#             --error=${log_path}/${model_name}_a0_k1.err \
#             run_job_validate_2fold.sh --model_name $model_name --num_heads 1 --alpha 0 --instruction_prompt $instruction_prompt --judge_name $judge_name --info_name $info_name
#     sleep 60

#     while [ $(squeue -u $user | awk '$5 ~ /^(R|PD)$/' | wc -l) -ge $max_jobs ]; do
#         echo "Maximum number of jobs ($max_jobs) reached. Waiting..."
#         sleep 60
#     done

#     sbatch --job-name=validate_2fold_${model_name}_a15_k48 \
#             --output=${log_path}/${model_name}_a15_k48.out \
#             --error=${log_path}/${model_name}_a15_k48.err \
#             run_job_validate_2fold.sh --model_name $model_name --num_heads 48 --alpha 15 --instruction_prompt $instruction_prompt --judge_name $judge_name --info_name $info_name
#     sleep 60

#     while [ $(squeue -u $user | awk '$5 ~ /^(R|PD)$/' | wc -l) -ge $max_jobs ]; do
#         echo "Maximum number of jobs ($max_jobs) reached. Waiting..."
#         sleep 60
#     done

#     sbatch --job-name=validate_2fold_honest_${model_name}_a0_k1 \
#             --output=${log_path}/honest_${model_name}_a0_k1.out \
#             --error=${log_path}/honest_${model_name}_a0_k1.err \
#             run_job_validate_2fold.sh --model_name $model_name --model_prefix local_ --num_heads 1 --alpha 0 --instruction_prompt $instruction_prompt --judge_name $judge_name --info_name $info_name
#     sleep 60
# done