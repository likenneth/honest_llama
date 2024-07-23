JUDGE=""
INFO=""
LOG_FILE="sweep_output.log"

for alpha in 10 15 20 25; do
    for K in 48 64 80 96; do
        echo "alpha: $alpha K: $K"
        python validate_2fold.py llama3_8B --num_heads $K --alpha $alpha --device 0 --num_fold 2 --use_center_of_mass --judge_name $JUDGE --info_name $INFO
        echo
        echo
    done
done