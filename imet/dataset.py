from pathlib import Path
from typing import Callable, List

import cv2
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .transforms import tensor_transform
from .utils import ON_KAGGLE


N_CLASSES = 1103
DATA_ROOT = Path('../input/imet-2019-fgvc6' if ON_KAGGLE else './data')


class TrainDataset(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame,
                 image_transform: Callable, debug: bool = True):
        super().__init__()
        self._root = root
        self._df = df
        self._image_transform = image_transform
        self._debug = debug

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        item = self._df.iloc[idx]
        image = load_transform_image(
            item, self._root, self._image_transform, debug=self._debug)
        target = torch.from_numpy(
            item.iloc[1:-1].values.astype("float32")).float()
        return image, target


class TestDataset(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame,
                 image_transform: Callable, debug: bool = True):
        self._root = root
        self._df = df
        self._image_transform = image_transform
        self._debug = debug

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        item = self._df.iloc[idx]
        image = load_transform_image(
            item, self._root, self._image_transform, debug=self._debug)
        return image, 0


class TTADataset:
    def __init__(self, root: Path, df: pd.DataFrame,
                 image_transform: Callable, tta: int):
        self._root = root
        self._df = df
        self._image_transform = image_transform
        self._tta = tta

    def __len__(self):
        return len(self._df) * self._tta

    def __getitem__(self, idx):
        item = self._df.iloc[idx % len(self._df)]
        image = load_transform_image(item, self._root, self._image_transform)
        return image, item.id


def load_transform_image(
        item, root: Path, image_transform: Callable, debug: bool = False):
    image = load_image(item, root)
    image = image_transform(image=image)["image"]
    if debug:
        image.save('_debug.png')
    tensor = tensor_transform(image=image)["image"]
    return tensor


def load_image(item, root: Path) -> Image.Image:
    image = cv2.imread(str(root / f'{item.id}.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # base_size = min(image.shape[0], image.shape[1])
    # ratio = 256 / base_size
    # image = cv2.resize(image, None, fx=ratio, fy=ratio,
    #                    interpolation=cv2.INTER_CUBIC)
    return image


def get_ids(root: Path) -> List[str]:
    return sorted({p.name.split('_')[0] for p in root.glob('*.png')})
