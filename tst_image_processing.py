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
import torch
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
    path = '/mnt/Datasets/kaggle_diabetic_retinopathy/0_original/train/4/'
    fname = '7531_right.jpeg'
    # path = '/home/md/Temp/'
    # fname = 'xxx.png'
    img = my_it.get_image(fname, path)

    # img=my_it.resize(img,512)
    img1 = my_it.autocrop(img, True)
    img1 = my_it.resize(img1, 512)
    cv.imshow('image1', img1)

    img2 = my_it.autocrop(img, False)
    img2 = my_it.resize(img2, 512)
    cv.imshow('image2', img2)
    cv.waitKey(0)
    cv.destroyAllWindows()
