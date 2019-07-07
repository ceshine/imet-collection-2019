import warnings
import argparse
from itertools import islice
import json
from pathlib import Path
import warnings
from typing import Dict, Callable, List
from functools import partial
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, log_loss
from sklearn.exceptions import UndefinedMetricWarning
import torch
from torch import nn, cuda
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from helperbot import (
    freeze_layers, TriangularLR, BaseBot, WeightDecayOptimizerWrapper,
    GradualWarmupScheduler, FBeta, LearningRateSchedulerCallback,
    MixUpCallback
)

from .adabound import AdaBound
from .models import get_seresnet_model, get_densenet_model, get_seresnet_partial_model
from .dataset import TrainDataset, TestDataset, get_ids, N_CLASSES, DATA_ROOT
from .transforms import get_train_transform, get_test_transform, cv2
from .utils import ON_KAGGLE
from .loss import FocalLoss

CACHE_DIR = Path('/tmp/imet' if ON_KAGGLE else './data/cache/')
CACHE_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR = Path('.' if ON_KAGGLE else './data/cache/')
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def make_loader(args, ds_class, root, df: pd.DataFrame, image_transform, drop_last=False, shuffle=False) -> DataLoader:
    return DataLoader(
        ds_class(root, df, image_transform, debug=args.debug),
        shuffle=shuffle,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=drop_last
    )


def opt_params(layer, learning_rate, final_lr):
    return {'params': layer.parameters(), 'lr': learning_rate, 'final_lr': final_lr}


def setup_differential_learning_rates(
        optimizer_constructor: Callable[[List[Dict]], torch.optim.Optimizer],
        layer_groups: List[nn.Parameter],
        lrs: List[float], final_lrs: List[float]) -> torch.optim.Optimizer:
    assert len(layer_groups) == len(
        lrs), f'size mismatch, expected {len(layer_groups)} lrs, but got {len(lrs)}'
    optimizer = optimizer_constructor(
        [opt_params(*p) for p in zip(layer_groups, lrs, final_lrs)])
    return optimizer


@dataclass
class ImageClassificationBot(BaseBot):
    checkpoint_dir: Path = CACHE_DIR / "model_cache/"
    log_dir: Path = MODEL_DIR / "logs/"

    def __post_init__(self):
        super().__post_init__()
        self.loss_format = "%.6f"
        self.metrics = (FBeta(step=0.05, beta=2, average="samples"),)
        self.monitor_metric = "fbeta"

    def extract_prediction(self, x):
        return x


def train_stage_one(args, model, train_loader, valid_loader, criterion):
    optimizer = WeightDecayOptimizerWrapper(
        torch.optim.Adam(model.parameters(), lr=2e-3),
        0.1
    )
    freeze_layers(model, [True, True, False])

    # stage 1
    n_steps = len(train_loader) // 2
    bot = ImageClassificationBot(
        model=model, train_loader=train_loader,
        val_loader=valid_loader, clip_grad=10.,
        optimizer=optimizer, echo=not ON_KAGGLE,
        criterion=criterion,
        avg_window=len(train_loader) // 10,
        callbacks=[
            LearningRateSchedulerCallback(TriangularLR(
                optimizer, 100, ratio=3, steps_per_cycle=n_steps))
        ],
        pbar=not ON_KAGGLE, use_tensorboard=False
    )
    bot.logger.info(bot.criterion)
    bot.train(
        n_steps,
        log_interval=len(train_loader) // 10,
        snapshot_interval=len(train_loader) // 4
    )
    bot.load_model(bot.best_performers[0][1])
    torch.save(bot.model.state_dict(), str(
        CACHE_DIR / f"stage1_{args.fold}.pth"))
    bot.remove_checkpoints(keep=0)


def train_stage_two(args, model, train_loader, valid_loader, criterion):
    n_steps = len(train_loader) * args.epochs
    optimizer = WeightDecayOptimizerWrapper(
        setup_differential_learning_rates(
            partial(
                torch.optim.Adam, weight_decay=0
                # AdaBound, weight_decay=0, gamma=1/5000, betas=(.8, .999)
                # torch.optim.SGD, momentum=0.9
            ), model, [1e-5, 1e-4, 3e-4], [1., 1., 1.]
        ), weight_decay=5e-2, change_with_lr=True)
    freeze_layers(model, [True, False, False])
    bot = ImageClassificationBot(
        model=model, train_loader=train_loader,
        val_loader=valid_loader, clip_grad=10.,
        optimizer=optimizer, echo=not ON_KAGGLE,
        criterion=criterion,
        avg_window=len(train_loader) // 15,
        callbacks=[
            LearningRateSchedulerCallback(
                # TriangularLR(
                #     optimizer, 100, ratio=4, steps_per_cycle=n_steps
                # )
                GradualWarmupScheduler(
                    optimizer, 100, len(train_loader) * 3,
                    after_scheduler=CosineAnnealingLR(
                        optimizer, n_steps - len(train_loader) * 3
                    )
                )
            ),
            MixUpCallback(alpha=0.2)
        ],
        pbar=not ON_KAGGLE, use_tensorboard=not ON_KAGGLE
    )
    bot.logger.info(bot.criterion)
    bot.model.load_state_dict(torch.load(
        CACHE_DIR / f"stage1_{args.fold}.pth"))

    # def snapshot_or_not(step):
    #     if step < 4000:
    #         if step % 2000 == 0:
    #             return True
    #     elif (step - 4000) % 1000 == 0:
    #         return True
    #     return False

    bot.train(
        n_steps,
        log_interval=len(train_loader) // 20,
        snapshot_interval=len(train_loader) // 2,
        # snapshot_interval=snapshot_or_not,
        early_stopping_cnt=args.early_stop,
        min_improv=1e-4,
        keep_n_snapshots=1
    )
    bot.load_model(bot.best_performers[0][1])
    bot.remove_checkpoints(keep=0)

    # Final model
    torch.save(bot.model, MODEL_DIR / f"final_{args.fold}.pth")
    # Failover (args + state dict)
    torch.save(
        [args.arch, bot.model.state_dict()],
        MODEL_DIR / f"failover_{args.arch}_{args.fold}.pth"
    )


def find_best_fbeta_threshold(truth, probs, beta=2, step=0.05):
    best, best_thres = 0, -1
    argsorted = probs.argsort(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UndefinedMetricWarning)
        for thres in np.arange(step, .5, step):
            current = fbeta_score(
                truth,
                binarize_prediction(
                    probs, thres, argsorted
                ).astype("int8"),
                beta=beta, average="samples")
            if current > best:
                best = current
                best_thres = thres
    return best, best_thres


def print_eval(truth, preds):
    best_score, threshold = find_best_fbeta_threshold(
        truth, preds, beta=2, step=0.01
    )
    print(f"f2: {best_score:.4f} @ threshold {threshold:.2f}")
    print(f"loss: {log_loss(truth, preds) / preds.shape[1]:.8f}")


def eval_model(args, valid_loaders: List[DataLoader]):
    model_dir = MODEL_DIR / args.model
    model = torch.load(str(model_dir / f"final_{args.fold}.pth"))
    model = model.cuda()
    bot = ImageClassificationBot(
        model=model, train_loader=None,
        val_loader=None, optimizer=None,
        echo=not ON_KAGGLE, criterion=None,
        pbar=not ON_KAGGLE, avg_window=100
    )
    tmp = []
    for valid_loader in valid_loaders:
        preds, truth = bot.predict(valid_loader, return_y=True)
        preds = torch.sigmoid(preds)
        tmp.append(preds.numpy())
    # print(np.mean(tmp, axis=0, keepdims=False).shape, preds.numpy().shape)
    final_preds = np.mean(tmp, axis=0, keepdims=False)
    print_eval(
        truth.numpy(),
        final_preds
    )
    if args.min_samples > 0:
        final_preds = mask_predictions(args, final_preds)
        print_eval(
            truth.numpy(),
            final_preds
        )


def predict_model(args, df: pd.DataFrame, loaders: List[DataLoader], name: str):
    model_dir = MODEL_DIR / args.model
    model = torch.load(str(model_dir / f"final_{args.fold}.pth"))
    model = model.cuda()
    bot = ImageClassificationBot(
        model=model, train_loader=None,
        val_loader=None, optimizer=None,
        echo=not ON_KAGGLE, criterion=None,
        pbar=not ON_KAGGLE, avg_window=100
    )
    tmp = []
    model_dir = MODEL_DIR / args.model
    for loader in loaders:
        preds = bot.predict(loader, return_y=False)
        preds = torch.sigmoid(preds)
        tmp.append(preds.numpy())
    final_preds = np.mean(tmp, axis=0, keepdims=False)
    # print(np.isnan(final_preds).sum())
    df_preds = pd.DataFrame(final_preds, index=df["id"].values)
    df_preds.to_pickle(CACHE_DIR / f"preds_{name}_{args.fold}.pkl")


def mask_predictions(args, preds):
    folds = pd.read_pickle(CACHE_DIR / 'folds.pkl')
    mask = folds.iloc[:, 1:-1].sum(axis=0).values < args.min_samples
    print(mask.shape, preds.shape)
    print(f"Masking {sum(mask)} labels...")
    preds[:, mask] = 0
    return preds


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'validate',
                         'predict_valid', 'predict_test'])
    arg('--batch-size', type=int, default=32)
    arg('--step', type=int, default=1)
    arg('--workers', type=int, default=2 if ON_KAGGLE else 4)
    arg('--tta', type=int, default=4)
    arg('--epochs', type=int, default=10)
    arg('--arch', type=str, default='seresnext50')
    arg('--min-samples', type=int, default=0)
    arg('--debug', action='store_true')
    arg('--limit', type=int)
    arg('--alpha', type=float, default=.5)
    arg('--gamma', type=float, default=.25)
    arg('--fold', type=int, default=0)
    arg('--model', type=str, default=".")
    arg('--early-stop', type=int, default=5)
    args = parser.parse_args()

    if args.mode in ("train", "validate", "predict_valid"):
        folds = pd.read_pickle(CACHE_DIR / 'folds.pkl')
        train_root = DATA_ROOT / 'train'
        train_fold = folds[folds['fold'] != args.fold]
        valid_fold = folds[folds['fold'] == args.fold]
        if args.limit:
            train_fold = train_fold[:args.limit]
            valid_fold = valid_fold[:args.limit]

    use_cuda = cuda.is_available()
    train_transform = get_train_transform(cv2.BORDER_REFLECT_101)
    test_transform = get_test_transform()
    if args.mode == 'train':
        if args.arch == 'seresnext50':
            model = get_seresnet_model(
                arch="se_resnext50_32x4d",
                n_classes=N_CLASSES, pretrained=True if args.mode == 'train' else False)
        elif args.arch == 'seresnext101':
            model = get_seresnet_model(
                arch="se_resnext101_32x4d",
                n_classes=N_CLASSES, pretrained=True if args.mode == 'train' else False)
        elif args.arch == 'seresnext50-partial':
            train_transform = get_train_transform(cv2.BORDER_CONSTANT)
            model = get_seresnet_partial_model(
                arch="se_resnext50_32x4d",
                n_classes=N_CLASSES, pretrained=True if args.mode == 'train' else False)
        elif args.arch.startswith("densenet"):
            model = get_densenet_model(arch=args.arch)
        # elif args.arch.startswith("efficientnet"):
        #     model = get_efficientnet(arch=args.arch)
        else:
            raise ValueError("No such model")
        if use_cuda:
            model = model.cuda()
        # criterion = nn.BCEWithLogitsLoss()
        criterion = FocalLoss(gamma=args.gamma, alpha=args.alpha)
        (CACHE_DIR / 'params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        train_loader = make_loader(
            args, TrainDataset, train_root, train_fold, train_transform, drop_last=True, shuffle=True)
        valid_loader = make_loader(
            args, TrainDataset, train_root, valid_fold, test_transform, shuffle=False)

        print(f'{len(train_loader.dataset):,} items in train, '
              f'{len(valid_loader.dataset):,} in valid')

        # Stage 1
        train_stage_one(args, model, train_loader, valid_loader, criterion)

        # Stage 2
        train_stage_two(args, model, train_loader, valid_loader, criterion)

    elif args.mode == 'validate':
        valid_loaders = [
            make_loader(
                args, TrainDataset, train_root,
                valid_fold, get_test_transform(), shuffle=False, drop_last=False),
            make_loader(
                args, TrainDataset, train_root,
                valid_fold, get_test_transform(flip=True), shuffle=False, drop_last=False)
        ]
        eval_model(args, valid_loaders)
    elif args.mode.startswith('predict'):
        if args.mode == 'predict_valid':
            loaders = [
                make_loader(
                    args, TestDataset, train_root,
                    valid_fold, get_test_transform(), shuffle=False, drop_last=False),
                make_loader(
                    args, TestDataset, train_root,
                    valid_fold, get_test_transform(flip=True), shuffle=False, drop_last=False)
            ]
            predict_model(args, valid_fold, loaders, "valid")
        elif args.mode == 'predict_test':
            test_root = DATA_ROOT / 'test'
            df_test = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
            if args.limit:
                df_test = df_test[:args.limit]
            print(df_test.shape)
            loaders = [
                make_loader(
                    args, TestDataset, test_root, df_test,
                    get_test_transform(), shuffle=False, drop_last=False),
                make_loader(
                    args, TestDataset, test_root, df_test,
                    get_test_transform(flip=True), shuffle=False, drop_last=False)
            ]
            predict_model(args, df_test, loaders, "test")


def binarize_prediction(probabilities, threshold: float, argsorted=None,
                        min_labels=1, max_labels=10):
    """ Return matrix of 0/1 predictions, same shape as probabilities.
    """
    assert probabilities.shape[1] == N_CLASSES
    if argsorted is None:
        argsorted = probabilities.argsort(axis=1)
    max_mask = _make_mask(argsorted, max_labels)
    min_mask = _make_mask(argsorted, min_labels)
    prob_mask = probabilities > threshold
    return (max_mask & prob_mask) | min_mask


def _make_mask(argsorted, top_n: int):
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = [i // top_n for i in range(len(col_indices))]
    mask[row_indices, col_indices] = 1
    return mask


if __name__ == '__main__':
    main()
