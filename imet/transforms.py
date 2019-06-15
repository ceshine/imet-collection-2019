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
from albumentations.pytorch.transforms import ToTensor

cv2.setNumThreads(0)

train_transform = Compose([
    # PadIfNeeded(256, 256, border_mode=cv2.BORDER_CONSTANT., value=0, p=1.),
    # ShiftScaleRotate(
    #     shift_limit=0.0625, scale_limit=0.1, rotate_limit=30,
    #     border_mode=cv2.BORDER_REFLECT_101, p=1.),
    RandomScale(scale_limit=0.125),
    Rotate(limit=20,  border_mode=cv2.BORDER_REFLECT_101, p=1.),
    OneOf([
        RandomCrop(256, 256, p=0.9),
        CenterCrop(256, 256, p=0.1),
    ], p=1.),
    HorizontalFlip(p=0.5),
    RandomContrast(limit=0.2, p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    RandomBrightness(limit=0.2, p=0.5),
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


test_transform = Compose([
    # SmallestMaxSize(320),
    RandomScale(scale_limit=0.125),
    # PadIfNeeded(256, 256, border_mode=cv2.BORDER_REFLECT_101, value=0, p=1.),
    OneOf([
        RandomCrop(256, 256, p=0.9),
        CenterCrop(256, 256, p=0.1),
    ], p=1.),
    HorizontalFlip(p=0.5),
])


tensor_transform = ToTensor(normalize=dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)
