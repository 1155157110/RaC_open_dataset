# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/rxavier/economicus

import copy
import datasets

# fold0
def get_preprocessed_custom(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("csv", data_files="../llama3.1-8b/RaC-Datasets/Datasets_17k_balance_10fold/train/augmented_x24_train_split/Datasets_17k_augmented_train_10fold_0.csv", split='train', encoding = "utf-8")

    if split == 'train':
       dataset = dataset.filter(lambda x, idx: idx < int(dataset.num_rows*0.95), with_indices=True)
    elif split == 'validation':
        dataset = dataset.filter(lambda x, idx: int(dataset.num_rows*0.95) < idx, with_indices=True)
    else:
        raise NotImplementedError

    def tokenize_add_label(sample):
        question = "Answer the question:\n" + sample['Question']+'\n'
        answer_rephrase = "Correct answer:\n"+sample['Correct Answer']+"\n"
        answer_rephrase += "Explanation:\n"+sample['Rephrase']+"\n"
        answer_explanation = answer_rephrase
        answer_explanation += "The explanation to the correct answer is: "+sample['Correct Answer Expl']+"\n"

        prompt = tokenizer.encode(tokenizer.bos_token + question, add_special_tokens=False)
        # response = tokenizer.encode(answer_rephrase +  tokenizer.eos_token, add_special_tokens=False)
        response = tokenizer.encode(answer_explanation +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + response,
            "attention_mask" : [1] * (len(prompt) + len(response)),
            "labels": [-100] * len(prompt) + response
        }
        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset
