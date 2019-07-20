# imet-collection-2019

A fairly generic solution to iMet Collection 2019 - FGVC6 on Kaggle

Credit: This solution is built upon [Konstantin Lopuhin's public baseline](https://github.com/lopuhin/kaggle-imet-2019).

## Requirements

Directly taken from [requirements.txt](requirements.txt) (they're also in [setup.py](setup.py)).

* torch>=1.0.0
* albumentations>=0.2.3
* pretrainedmodels>=0.7.4
* tqdm>=4.29.1
* scikit-learn>=0.21.2
* pandas>=0.24.0
* helperbot>=0.1.2

`helperbot` is included in this repo via `git subtree`. Install it after PyTorch and before everythin else:

```
cd pytorch_helper_bot && pip install .
```

## Environments

I trained all my models using Kaggle Kernel. Example public kernels can be found at:

* [Trainer](https://www.kaggle.com/ceshine/imet-trainer)
* [Validation (with TTA)](https://www.kaggle.com/ceshine/imet-validation-kernel-public)
* [Inference](https://www.kaggle.com/ceshine/imet-inference-kernel-public?scriptVersionId=16663008) - Private score *0.614* with 3 models (already in bronze range).

One drawback of Kaggle Kernel is that it's hard to control the version of PyTorch. My models trained during competition were trained with PyTorch 1.0, but the model cannot be loaded in the post-competition kernels due to this [compatibility issue](https://github.com/pytorch/pytorch/issues/20756). (The issue was fixed in the PyTorch master branch, but has not been released yet at the time of writing.)

To avoid this kind of hassles in the future, I started to keep two versions of trained model: one which contains fully pickled model using `torch.save(model, f'final_{fold}.pth')` to speed up experiment iteration; and one which has only model weights and the name of the architecture as a failover using `torch.save([args.arch, model.state_dict()], f'failover_{args.arch}_{fold}.pth')`.

### Freezing the first three (Resnet) layers

The 10th place solution suggested that only training the last two (Renset) layers is sufficient to get good accuracies. This technique allow us to training se-resnext101 models in Kaggle Kernel with 320x320 images faster. (Otherwise the models will be underfit and underperformed relative to se-resnext50).

The code that freezes the first three layers lives in the [*freezing*](https://github.com/ceshine/imet-collection-2019/tree/freezing) branch.

* [3-model se-resnext101 Inference](https://www.kaggle.com/ceshine/imet-inference-kernel-public?scriptVersionId=17497470) - private 0.625
* [8-model se-resnext101 Inference](https://www.kaggle.com/ceshine/imet-inference-kernel-public?scriptVersionId=17498665) - private 0.629 (near silver range)

## Instructions

### Making K-Fold validation sets

Example:

```
python -m imet.make_folds --n-folds 10
```

This will create a `folds.pkl` that you can reuse later.

### Training model

Example:

```
python -m imet.main train --batch-size 48 --epochs 11 --fold 0 --arch seresnext101 --early-stop 4
```

### Evaluating model (with TTA)

Example:

```
python -m imet.main validate --fold 0 --batch-size 256 --tta 4 --model .
```

### Making Predictions (with TTA)

Example:

```
python -m imet.main predict_test --batch-size 256 --fold 0 --tta 5 --model ./seresnext50/
```

Then create a submission file (this one only uses predictions from three models):

```
python -m imet.make_submission test_0 test_1 test_2 --threshold 0.09
```