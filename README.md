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
