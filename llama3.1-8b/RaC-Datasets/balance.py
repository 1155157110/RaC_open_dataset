import random
import csv
import re
import pandas as pd
import json
import argparse

beginning = ['A)', 'B)', 'C)', 'D)']
random.seed(43)


data_dict = {}

def match_choice(match):
    return match[slice(*re.search(re.compile(r"[ABCD]"), match).span())]

def extract_question_and_answers(question_text):
    lines = question_text.split('\n')
    question_lines = []
    answer_lines = []
    answer_start = False

    for line in lines:
        if re.match(r'^[A-D]\)', line):  
            answer_lines.append(line.strip(' -\t'))
            answer_start = True
        elif answer_start:
            answer_lines.append(line.strip(' -\t'))
        else:
            question_lines.append(line)

    question = '\n'.join(question_lines)
    return question, answer_lines

def valid_question_answers(question,question_answers) -> bool:
    # print(f"{question_answers = }")
    if len(question_answers) != 4: 
        print(question_answers)
        print("Incorrect list length. Enter four new list elements in JSON format (e.g., ['A) ...', 'B) ...', 'C) ...', 'D) ...']).")
        user_input = input()
        new_list = json.loads(user_input)
        # import pdb
        # pdb.set_trace()
        if len(new_list) == 4:
            question_answers[:] = new_list 
            print("Revised list: ", question_answers)
            return True

    for i in range(4):
        if not question_answers[i].startswith(beginning[i]):
            # import pdb
            # pdb.set_trace()
            print(question)
            print(question_answers)
            print(f"option {beginning[i]} is incorrect. Enter the correct content: ")
            new_option = input()
            question_answers[i] = f"{beginning[i]}{new_option}"  
            print("Revised option: ", question_answers[i])

    return True

def replace_choice(prompt, correct_choice, selected_correct_choice):
    if correct_choice == selected_correct_choice:
        raise Exception("Replacing with same choice!")

    if correct_choice == 'A':
        pattern = re.compile(r'(?:[^a-zA-Z]|^)A\)')
        matches = pattern.findall(prompt)
        if len(matches) > 1:
            print(f"[A] Error: Too many choices found in prompt:\n")
            print(prompt)

        elif len(matches) == 1:
            index_found = prompt.index(matches[0])
            choice_pos = matches[0].index(match_choice(matches[0]))
            prompt_replaced = prompt[:index_found+choice_pos]+selected_correct_choice+prompt[index_found+choice_pos+1:]
            # global a_cnt
            # a_cnt += 1
            # print(f"Match prompt:\n\n{prompt}\n")
            return prompt_replaced
        else:
            return prompt

    elif correct_choice in "BCD":
        pattern = re.compile(r'(?:[^a-zA-Z]|^)[BCD](?:[^a-zA-Z]|$)')
        matches = pattern.findall(prompt)
        if len(matches) > 1:
            for m in matches:
                c = match_choice(m)
            return prompt
        elif len(matches) == 1:
            index_found = prompt.index(matches[0])
            choice_pos = matches[0].index(match_choice(matches[0]))
            return prompt[:index_found+choice_pos]+selected_correct_choice+prompt[index_found+choice_pos+1:]
        else:
            # print(f"[BCD] No answer found in prompt:\n\n{prompt}\n")
            # action = input("Enter an action to execute (empty for continue, x for stop): ")
            # if action.lower() == 'x': exit(-1)
            return prompt
    else:
        print(f"Error: invalid correct choice: {correct_choice}\nprompt:\n{prompt}")
        exit(-1)

def balance_dataset(input_path: str, output_path: str):
    df = pd.read_csv(input_path, engine='python')
    balanced_data = []
    print(f"Original dataset size: {len(df)}")

    for i in range(len(df)):
        # correct answer validation
        correct_answer = df['Correct Answer'][i].strip(' \n\t')     # answer: C) xxxx
        correct_choice = correct_answer[:2]                         # choice: C)
        assert correct_choice in beginning

        # question only prompt and question choice prompt
        question = df['Question'][i].strip('\n').split('\n')
        while '' in question:
            question.remove('')
        question = '\n'.join(question)
        question_answers = question.split('\n')[-4:]
        question_answers = [q.strip(' -\t') for q in question_answers]
        question = '\n'.join(question.split('\n')[:-4])
        assert valid_question_answers(question, question_answers)

        # randomly select a choice index to be the correct answer
        prompt_correct_answer = ""
        prompt_correct_answer_expl = df['Step_Two'][i]
        prompt_wrong_answers = [
            ans for ans in question_answers 
            if not ans.startswith(correct_choice)
        ]
        prompt_wrong_answers_expl = [
            df[f"Wrong Answer_{wa_idx}"][i] 
            for wa_idx in range(1, 4)
        ]
        selected_correct_choice = random.choice(beginning)

        # set the wrong answer and explanation indices
        if selected_correct_choice == correct_choice:
            prompt_correct_answer = correct_answer
        else:
            prompt_correct_answer = selected_correct_choice + correct_answer[2:]
            prompt_correct_answer_expl = replace_choice(
                prompt_correct_answer_expl, 
                correct_choice[0], 
                selected_correct_choice[0]
            )
            for idx, ans in enumerate(prompt_wrong_answers):
                if ans.startswith(selected_correct_choice):
                    prompt_wrong_answers[idx] = correct_choice + ans[2:]
            for idx, expl in enumerate(prompt_wrong_answers_expl):
                if expl.startswith(selected_correct_choice):
                    prompt_wrong_answers_expl[idx] = (
                        correct_choice + expl[2:]
                    )

        prompt_wrong_answers_expl.sort()
        prompt_answers = '\n'.join(
            sorted([prompt_correct_answer] + prompt_wrong_answers)
        )
        prompt_question = question + '\n' + prompt_answers
        prompt_rephrase = df['Step_One'][i]

        balanced_data.append([
            prompt_question,
            prompt_correct_answer,
            prompt_rephrase,
            prompt_correct_answer_expl,
            prompt_wrong_answers_expl[0],
            prompt_wrong_answers_expl[1],
            prompt_wrong_answers_expl[2]
        ])

    random.shuffle(balanced_data)
    balanced_df = pd.DataFrame(
        balanced_data,
        columns=[
            'Question', 'Correct Answer', 'Rephrase', 
            'Correct Answer Expl', 'Wrong Answer_1', 
            'Wrong Answer_2', 'Wrong Answer_3'
        ]
    )

    balanced_df.to_csv(output_path, encoding="utf-8", index=False)
    print(f"Balanced dataset saved to: {output_path}")
    print("check passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Balance multiple-choice dataset by shuffling correct answers."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="Datasets_17k.csv",
        help="original dataset path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="Datasets_17k_balanced.csv",
        help="balanced dataset path"
    )
    args = parser.parse_args()

    balance_dataset(args.input, args.output)