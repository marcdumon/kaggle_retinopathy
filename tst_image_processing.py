# --------------------------------------------------------------------------------------------------------
# 2019/02/27
# retinopathy - tst_image_processing.py
# md
# --------------------------------------------------------------------------------------------------------

import multiprocessing
from pathlib import Path
from pprint import pprint
from typing import Union, List

import cv2 as cv
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
    path = '/mnt/Datasets/kaggle_diabetic_retinopathy/0_original/train/0/'
    fname = '13_left.jpeg'
    # path = '/home/md/Temp/'
    # fname = 'ada.jpg'
    img = my_it.get_image(fname, path)
    cv.imwrite('x.jpg', img)

    # img=my_it.resize(img,512)
    img = my_it.random_pca(img, 64)

    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite('y.jpg', img)
