### Update 01/26/2024 :fire::fire:

[Zen](https://github.com/frankaging) provided this really cool library called [pyvene](https://github.com/stanfordnlp/pyvene) that can be used to load Inference-time Intervention, and many other mechanistic intervention technique. Here is what he says:

[pyvene](https://github.com/stanfordnlp/pyvene)  pushes for streamlining the sharing process of inference-time interventions and many more, comparing with other also super useful tools in this area!

I created the activation diff (~0.14MB) based on your shared LLaMA-2-chat by taking the bias terms. And your honest-llama can now be loaded as,

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyvene as pv

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.bfloat16,
).to("cuda")

pv_model = pv.IntervenableModel.load(
    # the activation diff ~0.14MB
    "zhengxuanzenwu/intervenable_honest_llama2_chat_7B", 
    model,
)

q = "What's a cure for insomnia that always works?"
prompt = tokenizer(q, return_tensors="pt").to("cuda")
_, iti_response_shared = pv_model.generate(
    prompt, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(iti_response_shared[0], skip_special_tokens=True))
```
This can be extended to other datasets easily I think by loading different activation diffs.

Huggingface repo is here: https://huggingface.co/zhengxuanzenwu/intervenable_honest_llama2_chat_7B. This intervention schema intervenes on every decoding step but not the prompt. But we can also do other things if needed.

Colab: https://colab.research.google.com/github/stanfordnlp/pyvene/blob/main/pyvene_101.ipynb#scrollTo=1c7b90b0

 
### Update 07/21/2023 :fire:

I edited a LLaMA-2 model to have baked-in intervention learnt from TruthfulQA dataset. The model is available [here](https://huggingface.co/likenneth/honest_llama2_chat_7B). You can load and play around starting from below:

```python
import torch
from pprint import pprint
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

model_name_new = "likenneth/honest_llama2_chat_7B"
tokenizer_new = AutoTokenizer.from_pretrained(model_name_new, trust_remote_code=True)
model_new = AutoModelForCausalLM.from_pretrained(model_name_new, low_cpu_mem_usage = True, torch_dtype=torch.float16, trust_remote_code=True)
_ = model_new.cuda()

q = "I ate a cherry seed. Will a cherry tree grow in my stomach?"
encoded_new = tokenizer_new(q, return_tensors = "pt")["input_ids"]
generated_new = model_new.generate(encoded_new.cuda())[0, encoded_new.shape[-1]:]
decoded_new = tokenizer_new.decode(generated_new, skip_special_tokens=True).strip()
pprint(decoded_new)
```
See `test.ipynb`.

# Honest LLaMA

This repository provides the code for the paper [Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://arxiv.org/abs/2306.03341). It shows how to apply **Inference-Time Intervention (ITI)** and various baseline methods to LLaMA, Alpaca and Vicuna.  

Some of the code is from [user-friendly llama](https://github.com/ypeleg/llama), thanks to Yam Peleg and Jason Phang. David Bau's [baukit](https://github.com/davidbau/baukit) comes in handy for implementing ITI, which we strongly recommend to anyone working on the internals of neural networks. [Kenneth Li](https://likenneth.github.io/) and [Oam Patel](https://github.com/0amp) made equal contributions to this work.  

## Abstract

> We introduce Inference-Time Intervention (ITI), a technique designed to enhance the truthfulness of large language models (LLMs). ITI operates by shifting model activations during inference, following a set of directions across a limited number of attention heads. This intervention significantly improves the performance of LLaMA models on the TruthfulQA benchmark. On an instruction-finetuned LLaMA called Alpaca, ITI improves its truthfulness from $32.5\%$ to $65.1\%$. We identify a tradeoff between truthfulness and helpfulness and demonstrate how to balance it by tuning the intervention strength. ITI is minimally invasive and computationally inexpensive. Moreover, the technique is data efficient: while approaches like RLHF require extensive annotations, ITI locates truthful directions using only few hundred examples. Our findings suggest that LLMs may have an internal representation of the likelihood of something being true, even as they produce falsehoods on the surface.

## Table of Contents
1. [Installation](#installation)
2. [TruthfulQA Evaluation](#truthfulqa-evaluation)
3. [Workflow](#workflow)
4. [How to Cite](#how-to-cite)


## Installation
In this the root folder of this repo, run the following commands to set things up.
```
conda env create -f environment.yaml
conda activate iti
python -m ipykernel install --user --name iti --display-name "iti"
mkdir -p validation/results_dump/answer_dump
mkdir -p validation/results_dump/summary_dump
mkdir -p validation/splits
mkdir features
git clone https://github.com/sylinrl/TruthfulQA.git
```

## TruthfulQA Evaluation

Since we need to evaluate using TruthfulQA API, you should first export your OpenAI API key as an environment variable. Then install following [their instructions](https://github.com/sylinrl/TruthfulQA). 

```
cd TruthfulQA/data
openai api fine_tunes.create -t finetune_truth.jsonl -m curie --n_epochs 5 --batch_size 21 --learning_rate_multiplier 0.1
openai api fine_tunes.create -t finetune_info.jsonl -m curie --n_epochs 5 --batch_size 21 --learning_rate_multiplier 0.1
cd ../..
```

If successful, you can find your GPT-judge and GPT-info model names with the command `openai api fine_tunes.list | grep fine_tuned_model`. It should be a string starting with `curie:ft-`.

## Workflow

(1) Get activations by running `bash get_activations.sh`. Layer-wise and head-wise activations are stored in the `features` folder. Prompts can be modified by changing the dataset-specific formatting functions in `utils.py`. 

(2) Get into `validation` folder, then, e.g., `CUDA_VISIBLE_DEVICES=0 python validate_2fold.py llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --judge_name <your GPT-judge name> --info_name <your GPT-info name>` to test inference-time intervention on LLaMA-7B. Read the code to learn about additional options.

(3) To create a modified model with ITI use `python edit_weight.py llama2_chat_7B` in the `validation` folder. `push_hf.py` can be used to upload this model to Huging Face.

**_NOTE:_** For a large model like `llama2_chat_70B` you may need to use multiple GPUs, so omit `CUDA_VISIBLE_DEVICES=0`. In addition, it may be beneficial to save the model locally first with `huggingface-cli download` and load with `--model_dir` options, availible in `get_activations.py`, `edit_weight.py` and `validate_2fold.py`.

### Results

See `results.md` for example result runs.

## Additional datasets

The modified nq_open and trivia_qa datasets used for transfer evaluation are available [here](https://huggingface.co/datasets/OamPatel/iti_nq_open_val) and [here](https://huggingface.co/datasets/OamPatel/iti_trivia_qa_val) respectively. 

## How to Cite

```
@misc{li2023inferencetime,
      title={Inference-Time Intervention: Eliciting Truthful Answers from a Language Model}, 
      author={Kenneth Li and Oam Patel and Fernanda Viégas and Hanspeter Pfister and Martin Wattenberg},
      year={2023},
      eprint={2306.03341},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
