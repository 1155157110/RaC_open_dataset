import os
import re
import argparse
import pandas as pd

choices = ['A', 'B', 'C', 'D']
ill_cnt = 0
answers = ""

'''

    python ./llama3.1-8b/get_analyse.py \
        --dataset Hard \
        --txt_path ./outputs/hard_result-3.txt


    python ./llama3.1-8b/get_analyse.py \
        --dataset Easy \
        --txt_path ./outputs/easy_result.txt

'''
    
def filter_prompt(prompt: str, pattern: re.Pattern):
    matches = pattern.findall(prompt)
    results = []
    for m in matches:
        candidate = m[0] if isinstance(m, tuple) else m
        candidate = candidate.strip().upper()
        if candidate in choices:
            results.append(candidate)

    unique = []
    for r in results:
        if r not in unique:
            unique.append(r)

    if len(unique) == 1:
        return unique[0]
    elif len(unique) > 1:
        return "X"
    return None


def get_choice(prompt: str) -> str:
    global ill_cnt
    prompt = prompt.strip()
    if len(prompt) == 1 and prompt.upper() in choices:
        return prompt.upper()
    if len(prompt) == 0:
        return "_"
    if len(prompt) >= 2 and prompt[0].upper() in choices and prompt[1] == '\n':
        return prompt[0].upper()

    patterns = [
        re.compile(r'[Tt]he\s+correct\s+answer\s+is\s*(?:option\s+)?\(?\s*([ABCD])\s*\)?'),
        re.compile(r'Correct\s+answer:\s*(?:option\s+)?\(?\s*([ABCD])\s*\)?'),
        re.compile(r'[Tt]he\s+answer\s+is\s*(?:option\s+)?\(?\s*([ABCD])\s*\)?'),
        re.compile(r'(?:(?:[Aa]?nswer)|swer):\s*(?:option\s+)?\(?\s*([ABCD])\s*\)?', re.IGNORECASE),
        re.compile(r'Answer:\s*(?:option\s+)?\(?\s*([ABCD])\s*\)?'),
        re.compile(r'^[\(\s]*([ABCD])[\)\.\s]', re.MULTILINE),
        re.compile(r'(?:[^A-Za-z]|^)\(?\s*([ABCD])\s*\)?(?:[^A-Za-z]|$)')
    ]

    for pattern in patterns:
        result = filter_prompt(prompt, pattern)
        if result and result != "":
            if result in choices:
                return result
            elif result == "X":
                continue
    ill_cnt += 1
    return "_"


def evaluate_prompt(prompt, expected_answer) -> int:
    global answers
    choice = get_choice(prompt)
    answers += choice
    return int(choice == expected_answer.upper())


def main(args):
    global ill_cnt
    dataset_name = args.dataset
    txt_path = args.txt_path
    dataset_path = 'llama3.1-8b/Data_OpenSource.xlsx'

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset Excel file not found: {dataset_path}")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"LLM output file not found: {txt_path}")

    df = pd.read_excel(dataset_path, sheet_name=dataset_name)
    correct_answers = [ans[0] for ans in df['Correct Answer']]
    total_count = len(correct_answers)

    correct_count = 0
    prompt_idx = 0
    current_prompt = ""

    with open(txt_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() == f"Prompt {prompt_idx}:":
                if prompt_idx > 0:
                    correct_count += evaluate_prompt(current_prompt, correct_answers[prompt_idx - 1])
                prompt_idx += 1
                current_prompt = ""
            else:
                current_prompt += line

        # handle last prompt
        if current_prompt.strip():
            correct_count += evaluate_prompt(current_prompt, correct_answers[prompt_idx - 1])

    accuracy = correct_count / total_count
    print(f"\nTotal: {total_count}:")
    print(f"{dataset_name:20} {correct_count} correct out of {total_count}, accuracy = {accuracy:.4f}")
    print(f"Unrecognized prompts (ill_cnt): {ill_cnt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=["Easy", "Hard", "ComprehensiveDataset"],
                        help="Sheet name from Data_OpenSource.xlsx")
    parser.add_argument('--txt_path', type=str, required=True,
                        help="Path to the .txt file containing LLM outputs")

    args = parser.parse_args()
    main(args)
