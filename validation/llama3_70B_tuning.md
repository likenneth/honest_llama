## Baseline
Compared to Llama 1 and Llama 2 models, Llama 3 does not have a baseline metric on TruthfulQA reported by Meta.

On Llama3_8B, we get:
```True*Info Score: 0.329, True Score: 0.459, Info Score: 0.716, MC1 Score: 0.302, MC2 Score: 0.490, CE Loss: 2.382, KL wrt Original: 0.0```

On Llama3_70B, we get:
```True*Info Score: 0.287, True Score: 0.758, Info Score: 0.379, MC1 Score: 0.340, MC2 Score: 0.518, CE Loss: 2.086, KL wrt Original: 0.0```

Surprisingly, the True*Info Score for Llama3_70B is lower than for Llama3_8B! The Info Score is the issue; the larger model refuses to answer questions much more often than the smaller model.
I hypothesize this is because the 70B model follows directions much more carefully, and because the prefix we attach to each question says to "reply “I have no comment” unless you are completely certain of the answer.",
the 70B model is much more conservative when choosing to answer questions. This is also probably why Meta did not release their TruthfulQA metrics on Llama3, since it performs worse than Llama2.

However, we only really care about the improvement obtained from ITI, so as long as we have Truth Score and Info Score not too extreme,
we can compare the effect of ITI from hyperparameter sweeping.

I removed the substring *"reply “I have no comment” unless you are completely certain of the answer"* from the instruction prompt and ran ``sweep_parallel.sh``:

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

We se that in every variation of ITI, we obtain 2-5% improvement in True*Info Score over the baseline: a modest improvement!