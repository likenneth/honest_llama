set -x
max_jobs=10
log_path="/n/holylfs06/LABS/kempner_undergrads/Users/jujipotle/honest_llama/get_activations/logs"

cd /n/holylfs06/LABS/kempner_undergrads/Users/jujipotle/honest_llama/get_activations

# model_names=("llama_7B" "llama2_chat_7B" "llama2_chat_13B" "llama2_chat_70B" "llama3_8B_instruct" "llama3_70B_instruct")
model_names=("llama2_chat_70B")
# Loop through model names
for i in "${!model_names[@]}"; do
    model_name=${model_names[$i]}
    sbatch --job-name=get_activations_${model_name}_tqa_mc2 \
           --output=${log_path}/${model_name}_tqa_mc2.out \
           --error=${log_path}/${model_name}_tqa_mc2.err \
           run_job_get_activations.sh --model_name $model_name --dataset_name tqa_mc2
    sleep 10
    sbatch --job-name=get_activations_${model_name}_tqa_gen \
           --output=${log_path}/${model_name}_tqa_gen.out \
           --error=${log_path}/${model_name}_tqa_gen.err \
           run_job_get_activations.sh --model_name $model_name --dataset_name tqa_gen_end_q
    sleep 10
done