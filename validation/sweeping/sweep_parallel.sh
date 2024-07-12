set -x
MAX_JOBS=4
LOG_PATH="/n/holylfs06/LABS/kempner_undergrads/Users/jujipotle/honest_llama/validation/sweeping"
MODEL="llama_8B"
for alpha in 10 15 20 25; do
    for K in 48 64 80 96; do
        while [ $(squeue -u $USER | awk '$5 ~ /^(R|PD)$/' | wc -l) -ge $MAX_JOBS ]; do
            echo "Maximum number of jobs ($MAX_JOBS) reached. Waiting..."
            sleep 60
        done
        sbatch --job-name=validate_2fold_${MODEL}_a${alpha}_k${K} \
               --output=${LOG_PATH}/${MODEL}_a${alpha}_k${K}.out \
               --error=${LOG_PATH}/${MODEL}_a${alpha}_k${K}.err \
               run_job_validate_2fold.sh --num_heads $K --alpha $alpha --model $MODEL
        sleep 60
    done
done