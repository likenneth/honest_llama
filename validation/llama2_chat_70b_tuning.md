## Baseline
Before looking for the best hyperparameters for applying ITI to the `llama2_chat_70B` we want to establish that the baseline model with `alpha` = 0 matches what Meta reportrd for the model!

According to https://huggingface.co/meta-llama/Llama-2-70b-chat-hf the percentage of generations that are both truthful and informative over the TruthfulQA dataset is *64.14%*.

We compare this with ```python validate_2fold.py llama2_chat_70B --num_heads 48 --alpha 0 --device 0 --num_fold 2 --model_dir <DIR> --use_center_of_mass --judge_name <curie:ft-...> --info_name <curie:ft-...>``

```
FOLD 0
Metric           GPT-info acc  GPT-judge acc       MC1       MC2   CE Loss  KL wrt Orig
Model                                                                                  
llama2_chat_70B      0.784841       0.733496  0.344743  0.536856  2.139822          0.0
FOLD 1
Metric           GPT-info acc  GPT-judge acc       MC1       MC2   CE Loss  KL wrt Orig
Model
                                                                                  
llama2_chat_70B      0.784314       0.678922  0.401961  0.589754  2.233409          0.0
True*Info Score: 0.5540755827508845, True Score: 0.7062089505728942, Info Score: 0.7845774006424086, MC1 Score: 0.37335203029867203, MC2 Score: 0.5633051972154774, CE Loss: 2.18661566644907, KL wrt Original: 0.0
```

As you can see, is quite in line with *64.14%*, just like in the 7B model.

## Validation on the https://huggingface.co/likenneth/honest_llama2_chat_70B with --num_heads 48 --alpha 15

```python validate_2fold.py llama2_chat_70B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --model_dir <DIR> --use_center_of_mass --judge_name <curie:ft-...> --info_name <curie:ft-...>``

```
FOLD 0
Metric           GPT-info acc  GPT-judge acc       MC1       MC2   CE Loss  KL wrt Orig
Model                                                                                  
llama2_chat_70B      0.904645       0.623472  0.359413  0.537248  2.161687     0.053351
FOLD 1
Metric           GPT-info acc  GPT-judge acc       MC1       MC2   CE Loss  KL wrt Orig
Model                                                                                  
llama2_chat_70B      0.843137       0.627451  0.392157  0.583664  2.251657     0.074485

True*Info Score: 0.5465853446663878, True Score: 0.6254614315163718, Info Score: 0.8738913658372884, MC1 Score: 0.3757850328395417, MC2 Score: 0.5604559422561715, CE Loss: 2.206671741604805, KL wrt Original: 0.06391817056573929
```

## Hyperparameter tunning

From https://github.com/likenneth/honest_llama/pull/24#issuecomment-1838847861
>As seen in results.md, the llama2-chat-70B is worse than the intervened llama2-chat-7B. My guess for this is that the 70B model has 80 layers and 64 heads per layer (compared to 32, 32, in 7B), so the hyper-parameters used for 7B are too small for 70B.

Therfore, we will do a hyperparameter sweep, similar to Fig4 of https://arxiv.org/pdf/2306.03341.pdf, focusing on higher values of K, which are likely better for the 70b model.

## Validation on the https://huggingface.co/likenneth/honest_llama2_chat_70B with --num_heads and --alpha sweep

## Sweep alpha 15, K [48, 64, 80, 96]
```
python validate_2fold.py llama2_chat_70B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --model_dir <DIR> --use_center_of_mass --judge_name <curie:ft-...> --info_name <curie:ft-...>

True*Info Score: 0.5465853446663878, True Score: 0.6254614315163718, Info Score: 0.8738913658372884, MC1 Score: 0.3757850328395417, MC2 Score: 0.5604559422561715, CE Loss: 2.206671741604805, KL wrt Original: 0.06391817056573929
```
```
python validate_2fold.py llama2_chat_70B --num_heads 64 --alpha 15 --device 0 --num_fold 2 --model_dir <DIR> --use_center_of_mass --judge_name <curie:ft-...> --info_name <curie:ft-...>

True*Info Score: 0.5205856548241125, True Score: 0.5850442255141666, Info Score: 0.889822738386308, MC1 Score: 0.37333704875593265, MC2 Score: 0.5572152026040502, CE Loss: 2.203287310600281, KL wrt Original: 0.08296061750501395
```
```
python validate_2fold.py llama2_chat_70B --num_heads 80 --alpha 15 --device 0 --num_fold 2 --model_dir <DIR> --use_center_of_mass --judge_name <curie:ft-...> --info_name <curie:ft-...>

True*Info Score: 0.5239537085765537, True Score: 0.5777002732633396, Info Score: 0.9069646195886667, MC1 Score: 0.3708920609808716, MC2 Score: 0.559189470883002, CE Loss: 2.210848189294338, KL wrt Original: 0.11165232319384813
```
```
python validate_2fold.py llama2_chat_70B --num_heads 96 --alpha 15 --device 0 --num_fold 2 --model_dir <DIR> --use_center_of_mass --judge_name <curie:ft-...> --info_name <curie:ft-...>

True*Info Score: 0.5288833711555987, True Score: 0.5862966824871758, Info Score: 0.9020746440385445, MC1 Score: 0.36845306582290616, MC2 Score: 0.559348046796374, CE Loss: 2.217066353857517, KL wrt Original: 0.1316926584765315
```
## Sweep alpha [20 25 30], K [48, 64, 80, 96]
To automate the process `sweep.sh` script was used. Results (copied from `sweep.log`):

