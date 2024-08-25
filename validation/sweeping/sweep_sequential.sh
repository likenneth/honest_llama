log_path="/path/to/honest_llama/validation/sweeping/logs/sweep_sequential.log"
model_name="llama_7B"
model_prefix=""
instruction_prompt="default"
judge_name="???"
info_name="???"

for alpha in 10 15 20; do
    for K in 48 64 80; do
        echo "alpha: $alpha K: $K"
        if [ -z "$model_prefix" ]; then
            python validate_2fold.py --model_name $model_name --num_heads $k --alpha $alpha --instruction_prompt $instruction_prompt --device 0 --num_fold 2 --use_center_of_mass --judge_name $judge_name --info_name $info_name
        else
            python validate_2fold.py --model_name $model_name --model_prefix $model_prefix --num_heads $k --alpha $alpha --instruction_prompt $instruction_prompt --device 0 --num_fold 2 --use_center_of_mass --judge_name $judge_name --info_name $info_name
        fi
    done
done