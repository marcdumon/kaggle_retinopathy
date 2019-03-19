# --------------------------------------------------------------------------------------------------------
# 2019/02/27
# retinopathy - test_image_processing.py
# md
# --------------------------------------------------------------------------------------------------------

import multiprocessing
from pathlib import Path
from pprint import pprint
from typing import Union, List

from PIL import Image
from numpy.random import RandomState
from pandas import DataFrame, read_csv, concat
from sacred import Experiment
from sacred.observers import MongoObserver
from multiprocessing import Pool
# from build_data import create_train_dataset
from my_toolbox import MyOsTools as my_ot
from my_toolbox import MyLogTools as my_lt
from my_toolbox import MyImageTools as my_it
from ruamel_yaml import YAML  # as yaml
import numpy as np
import matplotlib.pyplot as plt


def create_image(type: str = 'rgb', size: int = 128):
    im_arr = np.zeros((size, size, 3)).astype(np.uint8)
    if type == 'rgb':
        s = int(size / 3)
        im_arr[:s, :, 0] = 125  # Red
        im_arr[s:2 * s, :, 1] = 200  # Green
        im_arr[2 * s:3 * s + 2, :, 2] = 253  # Blue
    im = Image.fromarray(im_arr)
    return {'image': im, 'im_array': im_arr}


if __name__ == '__main__':

    create_new_image = 1

    # path = Path('/mnt/Datasets/kaggle_diabetic_retinopathy/0_original/train/0/')
    # fname = '13_left.jpeg'
    path = Path('/home/md/Temp/')
    fname = 'ada.jpg'

    img = None
    if create_new_image:
        img = create_image()['image']

    get_dataset_image = 1 - create_new_image
    if get_dataset_image:
        img = my_it.get_image(fname, path)

    im_ar = my_it.image_to_array(img)
    print(im_ar.shape)
    im_ar = my_it.minmax(im_ar, True)
    print(im_ar.shape)
    # im_ar = my_it.stdize(im_ar, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    im_ar = my_it.stdize(im_ar, mean=[0.1, 0., 0.], std=[1, 1, 1])
    print(im_ar.min(), im_ar.max(), 'ooooooooo')
    print('-' * 120)
    print(im_ar[..., 0])
    print('-' * 120)
    print(im_ar[..., 1])
    print('-' * 120)
    print(im_ar[..., 2])
    print('-' * 120)

    im_ar = im_ar * 255
    im_ar = im_ar.astype(np.uint8)  # Loose information
    img2 = Image.fromarray(im_ar)

    img.show('ORIGINAL')
    img2.show('PROCESSED')
    print(im_ar.max(), '------------')
