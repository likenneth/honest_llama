
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
import sys
sys.path.append('../')
# from utils import get_llama_activations_bau, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q

import llama
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Specific pyvene imports
from pv_utils import get_llama_activations_pyvene, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q
from interveners import wrapper, Collector, ITI_Intervener
import pyvene as pv

HF_NAMES = {
    # 'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'llama_7B': 'huggyllama/llama-7b',
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    # 'llama3_8B_instruct': '/n/holylfs06/LABS/kempner_undergrads/Users/jujipotle/llama3_8B_instruct_original/original',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct'
}

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    # tokenizer = llama.LlamaTokenizer.from_pretrained(model_name_or_path)
    # model = llama.LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
    device = "cuda"

    if args.dataset_name == "tqa_mc2": 
        dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")['validation']
        formatter = tokenized_tqa
    elif args.dataset_name == "tqa_gen": 
        dataset = load_dataset("truthfulqa/truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen
    elif args.dataset_name == 'tqa_gen_end_q': 
        dataset = load_dataset("truthfulqa/truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen_end_q
    else: 
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    if args.dataset_name == "tqa_gen" or args.dataset_name == "tqa_gen_end_q": 
        prompts, labels, categories = formatter(dataset, tokenizer)
        with open(f'../features/head_out_{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
    else: 
        prompts, labels = formatter(dataset, tokenizer)

    collectors = []
    pv_config = []
    for layer in range(model.config.num_hidden_layers): 
        collector = Collector(multiplier=0, head=-1) #head=-1 to collect all head activations, multiplier doens't matter
        collectors.append(collector)
        pv_config.append({
            # Unsure whether this is o_proj.input or v_proj.output
            # "component": f"model.layers[{layer}].self_attn.o_proj.input",
            "component": f"model.layers[{layer}].self_attn.v_proj.output",
            "intervention": wrapper(collector),
        })
    collected_model = pv.IntervenableModel(pv_config, model)
    # # tokenizer = llama.LlamaTokenizer.from_pretrained(model_name_or_path)
    # model = llama.LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
    device = "cuda"
    if args.dataset_name == "tqa_mc2": 
        dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")['validation']
        formatter = tokenized_tqa
    elif args.dataset_name == "tqa_gen": 
        dataset = load_dataset("truthfulqa/truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen
    elif args.dataset_name == 'tqa_gen_end_q': 
        dataset = load_dataset("truthfulqa/truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen_end_q
    else: 
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    if args.dataset_name == "tqa_gen" or args.dataset_name == "tqa_gen_end_q": 
        prompts, labels, categories = formatter(dataset, tokenizer)
        with open(f'../features/pyvene_{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
    else: 
        prompts, labels = formatter(dataset, tokenizer)

    all_layer_wise_activations = []
    all_head_wise_activations = []

    print("Getting activations")
    for prompt in tqdm(prompts):
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_pyvene(collected_model, collectors, prompt, device)
        all_layer_wise_activations.append(layer_wise_activations[:,-1,:].copy())
        all_head_wise_activations.append(head_wise_activations[:,-1,:].copy())

    print("Saving labels")
    np.save(f'../features/pyvene_{args.model_name}_{args.dataset_name}_labels.npy', labels)

    print("Saving layer wise activations")
    np.save(f'../features/pyvene_{args.model_name}_{args.dataset_name}_layer_wise.npy', all_layer_wise_activations)
    
    print("Saving head wise activations")
    np.save(f'../features/pyvene_{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations)

if __name__ == '__main__':
    main()
