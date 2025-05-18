import os
import pandas as pd
from sklearn.model_selection import KFold

K = 10

if __name__ == '__main__':

    dataset = pd.read_csv("Datasets_17k_balanced.csv")

    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
    dataset_size = len(dataset)

    output_folder = f"Datasets_17k_balance_{K}fold"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "train", "augmented_x4_train_split"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "train", "augmented_x24_train_split"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "train", "balanced_train_split"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "test"), exist_ok=True)

    kfold = KFold(n_splits=K, shuffle=True, random_state=43)

    fold = 0
    for train_split, test_split in kfold.split(dataset):
        train_data, test_data = dataset.iloc[train_split], dataset.iloc[test_split]
        train_data.to_csv(f"{output_folder}/train/balanced_train_split/Datasets_17k_balance_train_{K}fold_{fold}.csv", encoding='utf-8')
        test_data.to_csv(f"{output_folder}/test/Datasets_17k_balance_test_{K}fold_{fold}.csv", encoding='utf-8')
        fold += 1
