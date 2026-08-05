import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

hf_token = os.environ["HUGGINGFACE_TOKEN"]
model_id = '/scratch/wx2356/verilogFinetune/output/deepseek-coder-7b-finetune-nl2sva-machine'
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.push_to_hub('Jalik/deepseek-coder-7b-finetune-nl2sva-machine', private=True, token=hf_token)
tokenizer.push_to_hub('Jalik/deepseek-coder-7b-finetune-nl2sva-machine', private=True, token=hf_token)