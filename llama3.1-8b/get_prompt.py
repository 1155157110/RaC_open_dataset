import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer,AutoTokenizer

from peft import PeftModel, PeftConfig
import tqdm
import pandas as pd

'''

    python llama3.1-8b/get_prompt.py \
        --model_id "/home/a6000/Desktop/RAC_Journal/llama-3.1-8b/8B" \
        --lora_path "llama3.1-8b/RaC-Lora" \
        --dataset "Hard" \
        --output_file_path "./outputs/hard_result-no.txt"

    python llama3.1-8b/get_prompt.py \
        --model_id "/home/a6000/Desktop/RAC_Journal/llama-3.1-8b/8B" \
        --lora_path "llama3.1-8b/RaC-Lora" \
        --dataset "Easy" \
        --output_file_path "./outputs/easy_result-no.txt"

    python llama3.1-8b/get_prompt.py \
        --model_id "/home/a6000/Desktop/RAC_Journal/llama-3.1-8b/8B" \
        --lora_path "llama3.1-8b/RaC-Lora" \
        --dataset "Easy" \
        --output_file_path "./outputs/easy_result-no.txt"

'''

def get_response(eval_prompt, model, tokenizer):
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    model.eval()
    with torch.no_grad():
        model_output_tokens = model.generate(
            **model_input,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            do_sample = False
        )[0]
        model_output = tokenizer.decode(model_output_tokens, skip_special_tokens=True)
    return model_output[len(eval_prompt):]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--lora_path', type=str, required=True, help='Path to the LoRA weights')
    # parser.add_argument('--dataset_path', type=str, required=True, help='Path to the test dataset (xlsx or csv)')
    parser.add_argument('--dataset', type=str, choices=["Easy", "Hard", "Comprehensive"], required=True)
    parser.add_argument('--output_file_path', type=str, required=True, help='Output file path, including filename (e.g., ./output/result.txt)')

    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = LlamaForCausalLM.from_pretrained(args.model_id, device_map='auto', torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, args.lora_path)

    test_dataset = pd.read_excel('./llama3.1-8b/Data_OpenSource.xlsx', sheet_name=args.dataset)
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write responses
    with open(args.output_file_path, "w", encoding="utf-8") as output_file:
        for prompt_idx in tqdm.tqdm(range(len(test_dataset['Question']))):
            question = test_dataset['Question'][prompt_idx]
            prompt = f"Answer the question: {question}\n"

            response = get_response(
                prompt,
                model=model,
                tokenizer=tokenizer
            )
            output_file.write(f"Prompt {prompt_idx}:\n{response}\n")

    print(f"Responses written to {args.output_file_path}")


