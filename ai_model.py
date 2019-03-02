# --------------------------------------------------------------------------------------------------------
# 2019/03/01
# retinopathy - ai_model.py
# md
# --------------------------------------------------------------------------------------------------------

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

plt.ion()  # interactive mode
multiGPU = False

TRAIN_IMG_PATH = "../input/train"
TEST_IMG_PATH = "../input/test"
LABELS_CSV_PATH = "../input/labels.csv"
SAMPLE_SUB_PATH = "../input/sample_submission.csv"
