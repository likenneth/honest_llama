As part of a summer research project, I replicated ITI on Llama 2 and 3 models. Below are my results:

## Addressing Llama 3's "worse" performance on TruthfulQA compared to Llama 2
Compared to Llama 1 and Llama 2 models, Llama 3 does not have a baseline metric on TruthfulQA reported by Meta. So, we calculate the baseline ourselves:

On Llama3_8B, we get:
```True*Info Score: 0.329, True Score: 0.459, Info Score: 0.716, MC1 Score: 0.302, MC2 Score: 0.490, CE Loss: 2.382, KL wrt Original: 0.0```

On Llama3_70B, we get:
```True*Info Score: 0.287, True Score: 0.758, Info Score: 0.379, MC1 Score: 0.340, MC2 Score: 0.518, CE Loss: 2.086, KL wrt Original: 0.0```

Surprisingly, the True\*Info Score for Llama3_70B is lower than for Llama3_8B! The Info Score is the issue; from qualitative inspection of model responses, we observe that the larger model refuses to answer questions much more often than the smaller model. I hypothesize this is because the 70B model follows directions much more carefully, and because the prefix we attach to each question says to "reply 'I have no comment' unless you are completely certain of the answer.", the 70B model is much more conservative when choosing to answer questions. This is also probably why Meta did not release their TruthfulQA metrics on Llama3, since it performs 'worse' than Llama2, in terms of True\*Info Score. (We note that the True\*Info Score is not a one-size-fits-all metric, and that generally it is better to look at True Score and Info Score separately.)

However, we only really care about the improvement obtained from ITI, so as long as our Truth Score and Info Score are not too extreme,
we can compare the effect of ITI from hyperparameter sweeping.

I removed the substring *"reply “I have no comment” unless you are completely certain of the answer"* from the instruction prompt to obtain more informative answers and ran ``sweep_parallel.sh``. One can substitute other variations of instruction prompts to see their impact on the True*Info score.

```
| Model        | Intervention          | True*Info Score | True Score | Info Score | MC1 Score | MC2 Score | CE Loss | KL wrt Original |
|--------------|-----------------------|-----------------|------------|------------|-----------|-----------|---------|-----------------|
| llama3_70B   | Baseline              | 0.41            | 0.44       | 0.94       | 0.34      | 0.52      | 2.09    | Nan             |
| llama3_70B   | alpha: 10, heads: 48  | 0.43            | 0.45       | 0.95       | 0.35      | 0.53      | 2.10    | Nan             |
| llama3_70B   | alpha: 10, heads: 64  | 0.44            | 0.47       | 0.93       | 0.36      | 0.53      | 2.10    | Nan             |
| llama3_70B   | alpha: 10, heads: 80  | 0.45            | 0.49       | 0.93       | 0.36      | 0.53      | 2.10    | Nan             |
| llama3_70B   | alpha: 10, heads: 96  | 0.44            | 0.48       | 0.93       | 0.36      | 0.53      | 2.11    | Nan             |
| llama3_70B   | alpha: 15, heads: 48  | 0.43            | 0.45       | 0.95       | 0.36      | 0.53      | 2.11    | Nan             |
| llama3_70B   | alpha: 15, heads: 64  | 0.45            | 0.48       | 0.93       | 0.36      | 0.53      | 2.12    | Nan             |
| llama3_70B   | alpha: 15, heads: 80  | 0.46            | 0.50       | 0.92       | 0.36      | 0.54      | 2.13    | Nan             |
| llama3_70B   | alpha: 15, heads: 96  | 0.46            | 0.49       | 0.94       | 0.36      | 0.54      | 2.13    | Nan             |
| llama3_70B   | alpha: 20, heads: 48  | 0.45            | 0.47       | 0.95       | 0.35      | 0.53      | 2.13    | Nan             |
| llama3_70B   | alpha: 20, heads: 64  | 0.45            | 0.49       | 0.92       | 0.36      | 0.54      | 2.14    | Nan             |
| llama3_70B   | alpha: 20, heads: 80  | 0.47            | 0.52       | 0.90       | 0.36      | 0.54      | 2.16    | Nan             |
| llama3_70B   | alpha: 20, heads: 96  | 0.46            | 0.50       | 0.92       | 0.37      | 0.54      | 2.18    | Nan             |
| llama3_70B   | alpha: 25, heads: 48  | 0.44            | 0.47       | 0.95       | 0.35      | 0.53      | 2.16    | Nan             |
| llama3_70B   | alpha: 25, heads: 64  | 0.45            | 0.50       | 0.92       | 0.36      | 0.54      | 2.18    | Nan             |
| llama3_70B   | alpha: 25, heads: 80  | 0.46            | 0.51       | 0.90       | 0.36      | 0.54      | 2.21    | Nan             |
| llama3_70B   | alpha: 25, heads: 96  | 0.46            | 0.51       | 0.91       | 0.36      | 0.54      | 2.24    | Nan             |
```

We see that in every variation of ITI, we obtain 2-5 percentage points improvement in True*Info Score over the baseline: a modest improvement! In particular, each variation of ITI increases the True Score by as much as 8 percentage points, but only decreases the Info Score slightly, showing that ITI makes the model more truthful, without causing it to significantly refrain from answering questions.

Modifying the TruthfulQA instruction prompt the same way for Llama3_8B, we get:
```
| Model        | Intervention            | True*Info Score | True Score | Info Score | MC1 Score | MC2 Score | CE Loss | KL wrt Original |
|--------------|-------------------------|-----------------|------------|------------|-----------|-----------|---------|-----------------|
| llama3_8B    | Baseline                | 0.32            | 0.36       | 0.89       | 0.30      | 0.48      | 2.38    | 0.00            |
| llama3_8B    | alpha: 10, heads: 48    | 0.37            | 0.42       | 0.89       | 0.32      | 0.50      | 2.44    | 0.06            |
| llama3_8B    | alpha: 10, heads: 64    | 0.38            | 0.43       | 0.88       | 0.32      | 0.50      | 2.45    | 0.08            |
| llama3_8B    | alpha: 10, heads: 80    | 0.41            | 0.49       | 0.85       | 0.32      | 0.50      | 2.49    | 0.12            |
| llama3_8B    | alpha: 10, heads: 96    | 0.42            | 0.48       | 0.86       | 0.33      | 0.50      | 2.51    | 0.14            |
| llama3_8B    | alpha: 15, heads: 48    | 0.38            | 0.43       | 0.87       | 0.32      | 0.50      | 2.51    | 0.14            |
| llama3_8B    | alpha: 15, heads: 64    | 0.42            | 0.48       | 0.86       | 0.33      | 0.50      | 2.54    | 0.17            |
| llama3_8B    | alpha: 15, heads: 80    | 0.47            | 0.56       | 0.84       | 0.33      | 0.50      | 2.65    | 0.29            |
| llama3_8B    | alpha: 15, heads: 96    | 0.47            | 0.57       | 0.83       | 0.33      | 0.50      | 2.69    | 0.33            |
| llama3_8B    | alpha: 20, heads: 48    | 0.42            | 0.49       | 0.86       | 0.32      | 0.49      | 2.61    | 0.25            |
| llama3_8B    | alpha: 20, heads: 64    | 0.45            | 0.54       | 0.83       | 0.32      | 0.50      | 2.68    | 0.32            |
| llama3_8B    | alpha: 20, heads: 80    | 0.52            | 0.66       | 0.79       | 0.32      | 0.50      | 2.95    | 0.59            |
| llama3_8B    | alpha: 20, heads: 96    | 0.48            | 0.65       | 0.73       | 0.32      | 0.50      | 3.06    | 0.69            |
| llama3_8B    | alpha: 25, heads: 48    | 0.48            | 0.59       | 0.81       | 0.31      | 0.49      | 2.78    | 0.42            |
| llama3_8B    | alpha: 25, heads: 64    | 0.49            | 0.62       | 0.78       | 0.31      | 0.50      | 2.90    | 0.55            |
| llama3_8B    | alpha: 25, heads: 80    | 0.50            | 0.68       | 0.73       | 0.32      | 0.50      | 3.61    | 1.25            |
| llama3_8B    | alpha: 25, heads: 96    | 0.45            | 0.71       | 0.64       | 0.32      | 0.51      | 3.88    | 1.50            |
```

For each variation of ITI, the True Score increases significantly from the baseline, while the Info Score decreases slightly (with more degradation with heavy intervention, as expected).

In conclusion, I would recommend to people who wish to apply ITI to their model to measure the True Score and Info Score separately to see how much intervention needs to be applied to strike a good balance.

## Uploading baked-in ITI models to HuggingFace
I bake-in ITI interventions (with alpha=15, heads=48) into the following models: Llama_7B, Llama2_chat_7B, Llama2_chat_13B, Llama2_chat_70B, Llama3_8B_instruct, Llama3_70B_instruct. Notice that for some models, the baked-in ITI is stronger/weaker than applying ITI to the model directly. I hypothesize this is because the baked-in ITI indiscriminately applies ITI to every token of the input and the tokens generated after the prompt, while when we apply ITI directly, it is only applied to the tokens generated after the prompt. This may cause the baked-in ITI to have more of an influence on smaller models and less of an influence on larger models. I welcome further investigation into this phenonemon, please contact me if you have any ideas!

For now, I'd recommend using ITI directly to obtain more consistent results. But the baked-in models are all available in the HuggingFace collection [here](https://huggingface.co/collections/jujipotle/inference-time-intervention-iti-models-66ca15448347e21e8af6772e) for your convenience!

| Model                | Intervention | True*Info Score | True Score | Info Score | MC1 Score | MC2 Score | CE Loss | KL wrt Original |
|----------------------|--------------|-----------------|------------|------------|-----------|-----------|---------|-----------------|
| llama_7B             | Baseline     | 0.24            | 0.25       | 0.96       | 0.25      | 0.41      | 2.13    | 0.00            |
| llama_7B             | ITI          | 0.33            | 0.35       | 0.94       | 0.28      | 0.45      | 2.28    | 0.17            |
| llama_7B             | Baked-in     | 0.41            | 0.48       | 0.86       | 0.34      | 0.52      | 2.28    | 0.00            |
| llama2_chat_7B       | Baseline     | 0.51            | 0.64       | 0.79       | 0.34      | 0.51      | 2.47    | 0.00            |
| llama2_chat_7B       | ITI          | 0.53            | 0.66       | 0.80       | 0.35      | 0.52      | 2.51    | 0.06            |
| llama2_chat_7B       | Baked-in     | 0.50            | 0.64       | 0.78       | 0.33      | 0.51      | 2.46    | 0.00            |
| llama2_chat_13B      | Baseline     | 0.55            | 0.63       | 0.87       | 0.35      | 0.53      | 2.31    | 0.00            |
| llama2_chat_13B      | ITI          | 0.59            | 0.67       | 0.89       | 0.37      | 0.56      | 2.33    | 0.13            |
| llama2_chat_13B      | Baked-in     | 0.50            | 0.60       | 0.84       | 0.37      | 0.55      | 2.32    | 0.00            |
| llama2_chat_70B      | Baseline     | 0.45            | 0.63       | 0.71       | 0.37      | 0.56      | 2.19    | 0.00            |
| llama2_chat_70B      | ITI          | 0.46            | 0.62       | 0.74       | 0.38      | 0.57      | 2.18    | 0.01            |
| llama2_chat_70B      | Baked-in     | 0.47            | 0.62       | 0.75       | 0.38      | 0.57      | 2.19    | 0.00            |
| llama3_8B_instruct   | Baseline     | 0.54            | 0.64       | 0.84       | 0.39      | 0.59      | 2.78    | 0.00            |
| llama3_8B_instruct   | ITI          | 0.62            | 0.71       | 0.87       | 0.39      | 0.59      | 2.87    | 0.30            |
| llama3_8B_instruct   | Baked-in     | 0.55            | 0.76       | 0.72       | 0.38      | 0.58      | 2.85    | 0.00            |
| llama3_70B_instruct  | Baseline     | 0.32            | 0.81       | 0.40       | 0.44      | 0.64      | 2.49    | 0.00            |
| llama3_70B_instruct  | ITI          | 0.38            | 0.73       | 0.53       | 0.45      | 0.66      | 2.48    | 0.03            |
| llama3_70B_instruct  | Baked-in     | 0.38            | 0.67       | 0.57       | 0.42      | 0.63      | 2.48    | 0.00            |

-- Results contributed by Justin Ji @jujipotle.