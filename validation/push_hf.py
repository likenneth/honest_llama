import os
import sys
import argparse
sys.path.insert(0, "..")

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='model name')
    parser.add_argument('--model_path', type=str, required=True, help='path to the model')
    parser.add_argument('--username', type=str, required=True, help='username for push_to_hub')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    
    tokenizer.push_to_hub(f"{args.username}/honest_{args.model_name}")
    model.push_to_hub(f"{args.username}/honest_{args.model_name}")

if __name__ == "__main__":
    main()