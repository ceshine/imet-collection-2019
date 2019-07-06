import random
import math

import cv2
from PIL import Image
from torchvision.transforms import (
    Normalize, Compose, Resize)
from albumentations import (
    Compose, HorizontalFlip, Rotate, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, JpegCompression, GaussNoise,
    Cutout, MedianBlur, Blur, OneOf, IAAAdditiveGaussianNoise, OpticalDistortion,
    GridDistortion, IAAPiecewiseAffine, ShiftScaleRotate, CenterCrop,
    RandomCrop, CenterCrop, Resize, PadIfNeeded, RandomScale, SmallestMaxSize
)
import albumentations.augmentations.functional as F
from albumentations.pytorch.transforms import ToTensor

cv2.setNumThreads(0)


class RandomCropIfNeeded(RandomCrop):
    """Take from:
    https://www.kaggle.com/c/imet-2019-fgvc6/discussion/94687
    """

    def __init__(self, height, width, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply(self, img, h_start=0, w_start=0, **params):
        h, w, _ = img.shape
        return F.random_crop(img, min(self.height, h), min(self.width, w), h_start, w_start)


def get_train_transform(border_mode, size=320):
    return Compose([
        # PadIfNeeded(256, 256, border_mode=cv2.BORDER_CONSTANT., value=0, p=1.),
        # ShiftScaleRotate(
        #     shift_limit=0.0625, scale_limit=0.1, rotate_limit=30,
        #     border_mode=cv2.BORDER_REFLECT_101, p=1.),
        # RandomScale(scale_limit=0.125),
        # HorizontalFlip(p=0.5),
        # RandomContrast(limit=0.2, p=0.5),
        # RandomGamma(gamma_limit=(80, 120), p=0.5),
        # RandomBrightness(limit=0.2, p=0.5),
        # Rotate(limit=20,  border_mode=border_mode, p=1.),
        HorizontalFlip(p=0.5),
        OneOf([
            RandomBrightness(0.1, p=1),
            RandomContrast(0.1, p=1),
        ], p=0.3),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0,
                         rotate_limit=15, p=0.3),
        IAAAdditiveGaussianNoise(p=0.3),
        RandomCropIfNeeded(size * 2, size * 2),
        Resize(size, size),
        # HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
        #                    val_shift_limit=10, p=1.),
        # OneOf([
        #     OpticalDistortion(p=0.3),
        #     GridDistortion(p=0.1),
        #     IAAPiecewiseAffine(p=0.3),
        # ], p=0.2),
        # OneOf([
        #     IAAAdditiveGaussianNoise(
        #         loc=0, scale=(1., 6.75), per_channel=False, p=0.3),
        #     GaussNoise(var_limit=(5.0, 20.0), p=0.6),
        # ], p=0.5),
        # Cutout(num_holes=4, max_h_size=30, max_w_size=50, p=0.75),
        # JpegCompression(quality_lower=50, quality_upper=100, p=0.5)
    ])


def get_test_transform(size=320, flip=True):
    transformations = [
        # SmallestMaxSize(320),
        # RandomScale(scale_limit=0.125),
        # PadIfNeeded(256, 256, border_mode=cv2.BORDER_REFLECT_101, value=0, p=1.),
        # OneOf([
        #     RandomCrop(256, 256, p=0.9),
        #     CenterCrop(256, 256, p=0.1),
        # ], p=1.),
        RandomCropIfNeeded(size * 2, size * 2),
        Resize(size, size),
    ]
    if flip:
        transformations.append(HorizontalFlip(p=1.))
    return Compose(transformations)


tensor_transform = ToTensor(normalize=dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)
