# --------------------------------------------------------------------------------------------------------
# 2019/02/20
# retinopathy - xxx_sacred1.py
# md
# --------------------------------------------------------------------------------------------------------
import numpy as np
from sacred import Ingredient

data_ingredient = Ingredient('dataset')


@data_ingredient.config
def cfg():
    filename = 'my_dataset.npy'
    normalize = True


@data_ingredient.capture
def load_data(filename, normalize):
    data = np.load(filename)
    if normalize:
        data -= np.mean(data)
    return data
