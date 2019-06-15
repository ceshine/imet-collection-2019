import random
import argparse
from collections import defaultdict, Counter
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from .dataset import DATA_ROOT
from .main import CACHE_DIR
from .utils import ON_KAGGLE

N_CLASSES = 1103


def expand_labels():
    print("Expanding labels...")
    df = pd.read_csv(DATA_ROOT / 'train.csv')
    df_label_names = pd.read_csv(DATA_ROOT / "labels.csv")
    labels = np.zeros((len(df), N_CLASSES)).astype("uint8")
    for i, row in tqdm(df.iterrows(), total=df.shape[0], disable=ON_KAGGLE):
        for label in row['attribute_ids'].split(' '):
            labels[i, int(label)] = 1
    df_labels = pd.DataFrame(
        labels,
        index=df.index, columns=df_label_names.attribute_name.values
    )
    df = pd.concat([df[["id"]], df_labels], axis=1)
    df.to_pickle(str(CACHE_DIR / "train_expanded_labels.pickle"))
    return df


def make_folds(n_folds: int, min_occurence: int = 30) -> pd.DataFrame:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    try:
        df = pd.read_pickle(DATA_ROOT / "train_expanded_labels.pickle")
    except:
        df = expand_labels()
    skf = MultilabelStratifiedKFold(
        n_splits=n_folds, random_state=42, shuffle=True)
    print("Creating folds...")
    labels_to_use = (np.sum(df.iloc[:, 1:].values, axis=0) > min_occurence)
    empty_rows = np.sum(df.iloc[:, 1:].values[:, labels_to_use], axis=1) == 0
    print("Empty rows after truncating:", sum(empty_rows))
    print("Eligible labels:", sum(labels_to_use))
    df = df[~empty_rows]
    folds = np.array([-1] * len(df))
    for fold, (_, valid_idx) in enumerate(skf.split(df[["id"]], df.iloc[:, 1:].values[:, labels_to_use])):
        folds[valid_idx] = fold
    df['fold'] = folds
    return df


def make_folds_reference(n_folds: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / 'train.csv')
    cls_counts = Counter(cls for classes in df['attribute_ids'].str.split()
                         for cls in classes)
    fold_cls_counts: Dict = defaultdict(int)
    folds = [-1] * len(df)
    for item in tqdm(df.sample(frac=1, random_state=42, disable=ON_KAGGLE).itertuples(),
                     total=len(df)):
        cls = min(item.attribute_ids.split(), key=lambda cls: cls_counts[cls])
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts
                              if count == min_count])
        folds[item.Index] = fold
        for cls in item.attribute_ids.split():
            fold_cls_counts[fold, cls] += 1
    df['fold'] = folds
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=10)
    args = parser.parse_args()
    df = make_folds(n_folds=args.n_folds)
    df.to_pickle(CACHE_DIR / 'folds.pkl')


if __name__ == '__main__':
    main()
