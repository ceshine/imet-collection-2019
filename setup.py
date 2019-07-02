from setuptools import setup

setup(
    name='imet',
    packages=['imet'],
    install_requires=[
        'torch>=1.0.0',
        'albumentations>=0.2.3',
        'pretrainedmodels>=0.7.4',
        'pandas>=0.24.0',
        'scikit-learn>=0.21.2',
        'tqdm==4.29.1',
        'helperbot>=0.1.3'
    ]
)
