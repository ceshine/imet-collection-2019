import argparse

import pandas as pd

from .utils import mean_df
from .dataset import DATA_ROOT
from .main import binarize_prediction, CACHE_DIR


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('predictions', nargs='+')
    arg('--threshold', type=float, default=0.2)
    args = parser.parse_args()
    sample_submission = pd.read_csv(
        DATA_ROOT / 'sample_submission.csv', index_col='id')
    dfs = []
    for prediction in args.predictions:
        df = pd.read_pickle(
            CACHE_DIR / f"preds_{prediction}.pkl")
        print(df.shape)
        # print(df.isnull().sum().sum())
        df = df.reindex(sample_submission.index)
        print(df.isnull().sum().sum())
        dfs.append(df)
    df = pd.concat(dfs)
    df = mean_df(df)
    df[:] = binarize_prediction(df.values, threshold=args.threshold)
    df = df.apply(get_classes, axis=1)
    df.name = 'attribute_ids'
    df.to_csv("submission.csv", header=True)


def get_classes(item):
    return ' '.join(str(cls_idx) for cls_idx, is_present in item.items() if is_present)


if __name__ == '__main__':
    main()
