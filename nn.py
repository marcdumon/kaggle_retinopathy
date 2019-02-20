from pathlib import Path
import numpy as np
import pandas as pd

from fastai import *

path = Path('../data/smpl_train_512_rotation')

labels_df = pd.read_csv(path / 'labels.csv')
print(labels_df.groupby('label').agg('count'))
