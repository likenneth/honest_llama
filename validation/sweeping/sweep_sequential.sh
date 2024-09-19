log_path="/path/to/honest_llama/validation/sweeping/logs/sweep_sequential.log"
model_name="llama3_8B_instruct"
model_prefix=""
instruction_prompt="default"
judge_name="???"
info_name="???"

for alpha in 15; do
    for K in 48; do
        for seed in {1..10}; do
            echo "alpha: $alpha K: $K seed: $seed"
            if [ -z "$model_prefix" ]; then
                python validate_2fold.py --model_name $model_name --num_heads $K --alpha $alpha --instruction_prompt $instruction_prompt --device 0 --num_fold 2 --use_center_of_mass --judge_name $judge_name --info_name $info_name --seed $seed
            else
                python validate_2fold.py --model_name $model_name --model_prefix $model_prefix --num_heads $K --alpha $alpha --instruction_prompt $instruction_prompt --device 0 --num_fold 2 --use_center_of_mass --judge_name $judge_name --info_name $info_name --seed $seed
            fi
        done
    done
done