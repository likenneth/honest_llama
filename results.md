# Reproducing the paper

```CUDA_VISIBLE_DEVICES=0 python validate_2fold.py llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --judge_name <curie:ft-...> --info_name <curie:ft-...>```

```
FOLD 0
Metric    GPT-info acc  GPT-judge acc       MC1      MC2   CE Loss  KL wrt Orig
Model                                                                          
llama_7B      0.848411       0.427873  0.268949  0.42379  2.438083     0.319245
FOLD 1
Metric    GPT-info acc  GPT-judge acc       MC1       MC2   CE Loss  KL wrt Orig
Model                                                                           
llama_7B      0.921569       0.301471  0.252451  0.391318  2.506364     0.294378
True*Info Score: 0.3227307173440359, True Score: 0.36467172443549545, Info Score: 0.8849896926985953, MC1 Score: 0.2606998178244403, MC2 Score: 0.40755430185175656, CE Loss: 2.4722234553098676, KL wrt Original: 0.30681129746139046
```

# Examples of validation at apha=15 and K=48 for LLaMa Chat

```python validate_2fold.py llama2_chat_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --judge_name <curie:ft-...> --info_name <curie:ft-...>```

```
FOLD 0
Metric          GPT-info acc  GPT-judge acc       MC1       MC2   CE Loss  KL wrt Orig
Model                                                                                 
llama2_chat_7B       0.96088       0.894866  0.388753  0.591546  2.893812     0.743171
FOLD 1
Metric          GPT-info acc  GPT-judge acc       MC1      MC2   CE Loss  KL wrt Orig
Model                                                                                
llama2_chat_7B      0.848039       0.941176  0.416667  0.59892  2.866764     0.727213

True*Info Score: 0.8303130017427044, True Score: 0.9180209981303035, Info Score: 0.9044597056426482, MC1 Score: 0.40270986145069276, MC2 Score: 0.595233170710637, CE Loss: 2.8802879935503007, KL wrt Original: 0.7351922281086445
```

```python validate_2fold.py llama2_chat_13B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --judge_name <curie:ft-...> --info_name <curie:ft-...>```

```
FOLD 0
Metric           GPT-info acc  GPT-judge acc       MC1       MC2   CE Loss  KL wrt Orig
Model                                                                                  
llama2_chat_13B       0.97555       0.662592  0.337408  0.518094  2.499073     0.304115
FOLD 1
Metric           GPT-info acc  GPT-judge acc       MC1       MC2   CE Loss  KL wrt Orig
Model                                                                                  
llama2_chat_13B      0.946078       0.590686  0.387255  0.564319  2.486696     0.223035

True*Info Score: 0.6020836791355518, True Score: 0.6266389807756844, Info Score: 0.9608142768109689, MC1 Score: 0.3623316074596098, MC2 Score: 0.5412067197590136, CE Loss: 2.4928841513395312, KL wrt Original: 0.26357506241649387
```

```python validate_2fold.py llama2_chat_70B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --model_dir <DIR> --use_center_of_mass --judge_name <curie:ft-...> --info_name <curie:ft-...>```

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

**It appears that `True*Info` scales *down* with the model size! To explore this futher we start with the baseline:**

## Baseline at alpha=0

When we set `alpha` to 0, this effectively disables ITI and *should* match the "vanilla" models (i.e. https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)

This is what we *expect*:

- Llama-2-Chat-7B	**57.04%**
- Llama-2-Chat-13B	**62.18%**
- Llama-2-Chat-70B	**64.14%**

But we get slightly different `True*Info`:

- Llama-2-Chat-7B	**60.05%**
- Llama-2-Chat-13B	**63.62%**
- Llama-2-Chat-70B	**55.41%**
(see logs bellow)



```python validate_2fold.py llama2_chat_7B --num_heads 48 --alpha 0 --device 0 --num_fold 2 --use_center_of_mass --judge_name <curie:ft-...> --info_name <curie:ft-...>```

```
FOLD 0
Metric          GPT-info acc  GPT-judge acc       MC1       MC2   CE Loss  KL wrt Orig
Model                                                                                 
llama2_chat_7B      0.887531       0.701711  0.330073  0.510557  2.416013          0.0
FOLD 1
Metric          GPT-info acc  GPT-judge acc       MC1       MC2   CE Loss  KL wrt Orig
Model                                                                                 
llama2_chat_7B      0.840686       0.698529  0.343137  0.515294  2.517075          0.0

True*Info Score: 0.6049799761446524, True Score: 0.7001204516036243, Info Score: 0.8641084184284962, MC1 Score: 0.3366053022676063, MC2 Score: 0.5129253583570372, CE Loss: 2.4665440082550045, KL wrt Original: 0.0

```

```python validate_2fold.py llama2_chat_13B --num_heads 48 --alpha 0 --device 0 --num_fold 2 --use_center_of_mass --judge_name <curie:ft-...> --info_name <curie:ft-...>```

```
FOLD 0
Metric           GPT-info acc  GPT-judge acc       MC1       MC2   CE Loss  KL wrt Orig
Model                                                                                  
llama2_chat_13B      0.909535       0.682152  0.330073  0.498951  2.252217          0.0
FOLD 1
Metric           GPT-info acc  GPT-judge acc       MC1       MC2  CE Loss  KL wrt Orig
Model                                                                                 
llama2_chat_13B      0.914216       0.713235  0.377451  0.567272  2.36487          0.0

True*Info Score: 0.6362096043277299, True Score: 0.6976934416798504, Info Score: 0.911875569298624, MC1 Score: 0.35376216501270435, MC2 Score: 0.5331115737143017, CE Loss: 2.3085431772470475, KL wrt Original: 0.0
```

 ```python validate_2fold.py llama2_chat_70B --num_heads 48 --alpha 0 --device 0 --num_fold 2 --model_dir <DIR> --use_center_of_mass --judge_name <curie:ft-...> --info_name <curie:ft-...>```

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

While the baseline `True*Info` is different from what we might expect, it's sitll ~0.6, and for 13b and 70b they are *higher* than `alpha=15`!

Can this be due to wrong hyperparamenters (`alpah` and `K`)?

## Hyperparameter tuning

`sweep.sh` automates the process with ```nohup sweep.sh > sweep.log 2> sweep_err.log &```

See `llama2_chat_70b_tuning.md` for the hyperparameter sweep for the 70b LLama2-chat. The alpha=15, K=48 from the paper may not be the optimal choice for the large model with more attention heads.

# Conclusion

As can be seen from `llama2_chat_70b_tuning.md`, there are *no* values of `alpha` and `K` for which `meta-llama/Llama-2-70b-chat-hf` beats the baseline of `TruthfulQA`!
The same is true for `meta-llama/Llama-2-13b-chat-hf`. We unfortunately must conclude that ITI does not scale well beyond 7B!
