#!/usr/bin/env bash


# openslide via conda
conda install -c conda-forge openslide

# torch and torchvision with cuda
pip install --root-user-action=ignore torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# regular pip requirements
pip install --root-user-action=ignore -r pip_requirements.txt
