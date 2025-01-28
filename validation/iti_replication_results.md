# ITI Replication Results

As part of a summer research project, I replicated ITI on Llama 2 and 3 models. For consistency, I used the same intervention (alpha=15, heads=48) and averaged scores across seeds 1-3. Below are my results:

### Results for Llama2_chat_7B
| Intervention | True Score | Info Score | MC1 Score | MC2 Score | CE Loss | KL wrt Original |
|--------------|------------|------------|-----------|-----------|---------|-----------------|
| Baseline     | 0.58       | 0.79       | 0.34      | 0.51      | 2.51    | 0.00            |
| ITI          | 0.83       | 0.93       | 0.40      | 0.58      | 2.82    | 0.65            |
| Baked-in     | 0.74       | 0.82       | 0.43      | 0.62      | 2.59    | 0.00            |

### Results for Llama2_chat_13B
| Intervention | True Score | Info Score | MC1 Score | MC2 Score | CE Loss | KL wrt Original |
|--------------|------------|------------|-----------|-----------|---------|-----------------|
| Baseline     | 0.58       | 0.87       | 0.35      | 0.53      | 2.35    | 0.00            |
| ITI          | 0.51       | 0.94       | 0.36      | 0.55      | 2.50    | 0.31            |
| Baked-in     | 0.54       | 0.88       | 0.36      | 0.54      | 2.34    | 0.00            |

### Results for Llama3_8B_instruct
| Intervention | True Score | Info Score | MC1 Score | MC2 Score | CE Loss | KL wrt Original |
|--------------|------------|------------|-----------|-----------|---------|-----------------|
| Baseline     | 0.60       | 0.83       | 0.39      | 0.59      | 2.81    | 0.00            |
| ITI          | 0.80       | 0.74       | 0.41      | 0.61      | 3.49    | 1.08            |
| Baked-in     | 0.62       | 0.77       | 0.41      | 0.61      | 2.90    | 0.00            |

For each model, there is an increase in the MC1 and MC2 scores. For the smaller models (llama2_chat_7B, llama3_8B_instruct), the truth score also shows substantial improvement. However, the larger model (llama2_chat_13B) may require stronger intervention to achieve similar gains in its truth score. I welcome contributors to share their results from additional hyperparameter sweeping experiments!

## Uploading Baked-in ITI Models to HuggingFace

I bake-in ITI interventions (with alpha=15, heads=48) into the following models: Llama_7B, Llama2_chat_7B, Llama2_chat_13B, Llama2_chat_70B, Llama3_8B_instruct, and Llama3_70B_instruct. The baked-in models are all available in the HuggingFace collection [here](https://huggingface.co/collections/jujipotle/inference-time-intervention-iti-models-66ca15448347e21e8af6772e) for your convenience!

-- Results contributed by Justin Ji @jujipotle.