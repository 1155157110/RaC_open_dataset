from itertools import permutations as P
import random
import re
import pandas as pd
import argparse
import sys
import os

sys.path.append(".")
from balance import replace_choice, valid_question_answers, extract_question_and_answers

K = 10

beginning = ['A)', 'B)', 'C)', 'D)']
random.seed(43)

parser = argparse.ArgumentParser(description='Input 1 positive numbers: fold index')
parser.add_argument('fold_index', type=int, help='fold index')
args = parser.parse_args()
df = pd.read_csv(f"./Datasets_17k_balance_10fold/train/balanced_train_split/Datasets_17k_balance_train_{K}fold_{args.fold_index}.csv")

if __name__ == "__main__":
    balanced_data = []
    print(f"{len(df) = }")

    for i in range(len(df)):
        for selected_correct_choice in ['A)', 'B)', 'C)', 'D)']:
            # correct answer validation
            correct_answer = df['Correct Answer'][i].strip(' \n\t')     # answer: C) xxxx
            correct_choice = correct_answer[:2]                         # choice: C)
            assert correct_choice in beginning

            # question only prompt and question choice prompt
            question = df['Question'][i].strip('\n').split('\n')
            while ('' in question):
                question.remove('')
            question = '\n'.join(q for q in question)
            question, question_answers = extract_question_and_answers(question)
            assert valid_question_answers(question,question_answers)

            # randomly select a choice index to be the correct answer
            prompt_correct_answer = ""
            prompt_correct_answer_expl = df['Correct Answer Expl'][i]
            prompt_wrong_answers = [ ans for ans in question_answers if not ans.startswith(correct_choice) ]
            prompt_wrong_answers_expl = [ df[f"Wrong Answer_{wa_idx}"][i] for wa_idx in range(1, 4) ]

            # set the wrong answer and explanation indexs
            if selected_correct_choice == correct_choice: # if sampled new choice is the same as the original choice:
                prompt_correct_answer = correct_answer
            else:
                prompt_correct_answer = selected_correct_choice+correct_answer[2:]
                prompt_correct_answer_expl = replace_choice(prompt_correct_answer_expl, correct_choice[0], selected_correct_choice[0])
                for wa_idx, ans in enumerate(prompt_wrong_answers):
                    if ans.startswith(selected_correct_choice):
                        prompt_wrong_answers[wa_idx] = correct_choice+prompt_wrong_answers[wa_idx][2:]
                for wa_idx, expl in enumerate(prompt_wrong_answers_expl):
                    if expl.startswith(selected_correct_choice):
                        prompt_wrong_answers_expl[wa_idx] = correct_choice+prompt_wrong_answers_expl[wa_idx][2:]

            prompt_wrong_answers_expl.sort()
            prompt_answers = '\n'.join(sorted([prompt_correct_answer]+prompt_wrong_answers))
            prompt_question = question + '\n' + prompt_answers
            prompt_rephrase = df['Rephrase'][i]

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
    augmented_df = pd.DataFrame(
        balanced_data,
        columns=['Question', 'Correct Answer', 'Rephrase', 'Correct Answer Expl', 'Wrong Answer_1', 'Wrong Answer_2', 'Wrong Answer_3']
    )


    output_folder = "./Datasets_17k_balance_10fold/train/augmented_x4_train_split"
    os.makedirs(output_folder, exist_ok=True)
    augmented_df.to_csv(f"{output_folder}/Datasets_17k_augmented_x4_train_{K}fold_{args.fold_index}.csv", encoding="utf-8")

    print("check passed")
