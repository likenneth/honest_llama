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
conda env create -f environment.yml
conda activate iti
pip install git+https://github.com/davidbau/baukit
pip install git+https://github.com/google-research/bleurt
pip install git+https://github.com/sylinrl/TruthfulQA
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

(2) Get into `validation` folder, then, e.g., `CUDA_VISIBLE_DEVICES=0 python validate_2fold.py llama_7B --num_heads 48 --alpha 20 --device 0 --num_fold 2 --use_center_of_mass --judge_name <your GPT-judge name> --info_name <your GPT-info name>` to test inference-time intervention on LLaMA-7B. Read the code to learn about additional options. 

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