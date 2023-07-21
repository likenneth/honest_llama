import os
import sys
sys.path.insert(0, "..")

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from llama.configuration_llama import LLaMAConfig
from llama.tokenization_llama import LLaMATokenizer
from llama.modeling_llama import LLaMAModel, LLaMAForCausalLM

model_name = 'results_dump/llama2_chat_7B_seed_42_top_48_heads_alpha_15'
tokenizer_ori = LLaMATokenizer.from_pretrained(model_name)
model_ori = LLaMAForCausalLM.from_pretrained(model_name, low_cpu_mem_usage = True, torch_dtype=torch.float16)

LLaMAConfig.register_for_auto_class()
LLaMATokenizer.register_for_auto_class("AutoTokenizer")
LLaMAModel.register_for_auto_class("AutoModel")
LLaMAForCausalLM.register_for_auto_class("AutoModelForCausalLM")

tokenizer_ori.push_to_hub("likenneth/honest_llama2_chat_7B")
model_ori.push_to_hub("likenneth/honest_llama2_chat_7B")