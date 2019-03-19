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
import cv2

# import cv2.CV_HOUGH_GRADIENT


if __name__ == '__main__':

    for i_name in ['0/256_right', '0/307_right', '0/613_right', '0/1024_right', '0/1497_right', '0/3080_right',
                   '4/3064_right', '4/4909_right', '4/5304_right', '4/15149_right', '4/23692_right', '4/32148_right']:
        # img = cv2.imread('/mnt/Datasets/kaggle_diabetic_retinopathy/experiments/2048px_6000i_0bt_autocropXresize/train/' + i_name + '.png', cv2.IMREAD_GRAYSCALE)
        img = cv2.imread('/mnt/Datasets/kaggle_diabetic_retinopathy/experiments/2048px_6000i_0bt_autocropXresize/train/' + i_name + '.png')
        # img = cv2.imread('/mnt/Datasets/kaggle_diabetic_retinopathy/experiments/2048px_6000i_0bt_autocropXresize/train/0/289_left.png', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(src=img, dsize=(0, 0), fx=.3, fy=0.3)
        print(img.shape)
        # sift = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=81, edgeThreshold=100, sigma=1.6)
        sift = cv2.ORB_create()
        # kp = sift.detect(img)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        print(descriptors)
        img = cv2.drawKeypoints(img, keypoints, None)
        cv2.imwrite(i_name.replace('/', '_') + '.png', img)
        print(i_name)
        # show the output image
        cv2.imshow("output", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
