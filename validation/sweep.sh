for alpha in 15 20 25 30; do
    for K in 48 64 80 96; do
        echo "alpha: $alpha K: $K"
        python validate_2fold.py llama2_chat_70B --num_heads $K --alpha $alpha --device 0 --num_fold 2 --model_dir $MODEL_DIR --use_center_of_mass --judge_name $JUDGE --info_name $INFO
        echo
        echo
    done
done
