# --------------------------------------------------------------------------------------------------------
# 2019/03/01
# retinopathy - ai_model.py
# md
# --------------------------------------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

from PIL import Image
import matplotlib.pyplot as plt
import time
import os
import fastai.vision as fv
import fastai.metrics as fm

multiGPU = True
# plt.ion()  # interactive mode

torch.manual_seed(999)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(999)
torch.backends.cudnn.deterministic = True

path = Path('/mnt/Datasets/kaggle_diabetic_retinopathy/experiments/512px_6000i_2bt_autocropXresize/')
labels_df = pd.read_csv(path / 'labels.csv')
print(labels_df.head())
print(labels_df.groupby('label').agg('count'))

# batch size
bs = 60

data = fv.ImageDataBunch.from_df(path=path / 'train_flat/', df=labels_df, fn_col=5, label_col=2, bs=bs)
data.show_batch(rows=3, figsize=(6, 6))
plt.show()
data.normalize()

kappa = fv.KappaScore()
kappa.weights = "quadratic"

learn = fv.create_cnn(data, fv.models.resnet34, metrics=[fm.error_rate, kappa])

learn.fit_one_cycle(15)  # , max_lr=slice(1e-6,5e-2))
