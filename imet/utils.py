import os
from datetime import datetime
from pathlib import Path
from multiprocessing.pool import ThreadPool
from typing import Dict

import pandas as pd
from scipy.stats.mstats import gmean
import torch
from torch import nn
from torch.utils.data import DataLoader


ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ


def gmean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level=0).agg(lambda x: gmean(list(x)))


def mean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level=0).mean()
