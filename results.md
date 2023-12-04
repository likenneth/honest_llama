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

```CUDA_VISIBLE_DEVICES=0 python validate_2fold.py llama2_chat_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --judge_name <curie:ft-...> --info_name <curie:ft-...>```

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