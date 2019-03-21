# --------------------------------------------------------------------------------------------------------
# 2019/02/27
# retinopathy - tst_image_processing.py
# md
# --------------------------------------------------------------------------------------------------------
import math
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
    path = '/mnt/Datasets/kaggle_diabetic_retinopathy/0_original/train/0/'
    # fname = '13_left.jpeg'
    # fname = '13_right.jpeg'
    fname = '46_right.jpeg'
    # path = Path('/home/md/Temp/')
    # fname = 'ada.jpg'
    orig = cv2.imread(path + fname, 1)
    orig = cv2.resize(orig, (int(orig.shape[1] / 4), int(orig.shape[0] / 4)))
    output = orig.copy()

    width, height = orig.shape[:2]
    print(width, height)
    # # create tmp images
    # rrr = np.array([width, height, 1]).astype(np.uint8)
    # ggg = np.array([width, height, 1]).astype(np.uint8)
    # bbb = np.array([width, height, 1]).astype(np.uint8)
    processed = np.zeros([width, height, 1]).astype(np.uint8)


    # rrr = cv2.CreateImage((orig.width, orig.height), cv2.IPL_DEPTH_8U, 1)
    # ggg = cv2.CreateImage((orig.width, orig.height), cv2.IPL_DEPTH_8U, 1)
    # bbb = cv2.CreateImage((orig.width, orig.height), cv2.IPL_DEPTH_8U, 1)
    # processed = cv2.CreateImage((orig.width, orig.height), cv2.IPL_DEPTH_8U, 1)
    # storage = cv2.CreateMat(orig.width, 1, cv2.CV_32FC3)
    # storage = np.zeros([width, 1]).astype(np.float)

    # print(processed.shape, storage.shape)

    def channel_processing(channel):
        channel = cv2.adaptiveThreshold(channel, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                        thresholdType=cv2.THRESH_BINARY, blockSize=55, C=7)
        # mop up the dirt
        channel = cv2.dilate(channel, None, 1)
        channel = cv2.erode(channel, None, 1)
        return channel


    def inter_centre_distance(x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


    def colliding_circles(circles):
        for index1, circle1 in enumerate(circles):
            for circle2 in circles[index1 + 1:]:
                x1, y1, Radius1 = circle1[0]
                x2, y2, Radius2 = circle2[0]
                # collision or containment:
                if inter_centre_distance(x1, y1, x2, y2) < Radius1 + Radius2:
                    return True


    # def find_circles(processed, storage, LOW):
    def find_circles(processed, LOW):
        try:
            storage = cv2.HoughCircles(processed, cv2.HOUGH_GRADIENT, 2, 32.0, 30,
                                       LOW)  # , 0, 100) great to add circle constraint sizes.

            # print(storage)
            print(storage.shape)

        except:  # Any error in try endsup into a infinit loop
            LOW += 1
            print('try')
            find_circles(processed, LOW)

        circles = np.asarray(storage)
        print(circles.shape)
        print('number of circles:', len(circles))
        if colliding_circles(circles):
            LOW += 1
            storage = find_circles(processed, storage, LOW)
        print('c', LOW)
        return storage


    def draw_circles(storage, output):
        circles = np.asarray(storage)
        print(len(circles), 'circles found')
        for circle in circles:
            Radius, x, y = int(circle[0][2]), int(circle[0][0]), int(circle[0][1])
            cv2.circle(output, (x, y), 1, (0, 255, 0), -1, 8, 0)
            cv2.circle(output, (x, y), Radius, (255, 0, 0), 3, 8, 0)


    # split image into RGB components
    # cv2.Split(orig, rrr, ggg, bbb, None)
    rrr, ggg, bbb = cv2.split(orig)
    # process each component
    rrr = channel_processing(rrr)
    ggg = channel_processing(ggg)
    bbb = channel_processing(bbb)
    # combine images using logical 'And' to avoid saturation
    cv2.bitwise_and(rrr, ggg, rrr)
    cv2.bitwise_and(rrr, bbb, processed)
    cv2.imshow('before canny', processed)
    cv2.waitKey(0)
    # cv2.SaveImage('case3_processed.jpg',processed)
    # use canny, as HoughCircles seems to prefer ring like circles to filled ones.
    processed = cv2.Canny(processed, 5, 70, 3)
    # smooth to reduce noise a bit more
    # cv2.Smooth(processed, processed, cv2.CV_GAUSSIAN, 7, 7)
    cv2.imshow('after canny', processed)
    cv2.waitKey(0)

    processed = cv2.GaussianBlur(processed, (7, 7), 2)
    cv2.imshow('after gauss', processed)
    cv2.waitKey(0)

    # cv2.imshow('processed', processed)
    # find circles, with parameter search
    storage = find_circles(processed, 100)
    draw_circles(storage, output)
    # show images
    cv2.imshow("original with circles", output)
    cv2.imshow('case1.jpg', output)

    cv2.waitKey(0)
    '''
    orig = cv2.imread(path + fname, 1)
    orig = cv2.resize(orig, (int(orig.shape[1] / 3), int(orig.shape[0] / 3)))
    img = orig.copy()
    # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img=img[...,0]-img[...,1]-img[...,2]
    print(img.shape)
    # img_original = img.copy()

    d_red = (150, 55, 65)
    # l_red = cv2.CV_RGB(250, 200, 200)
    l_red = (250, 200, 200)

    detector = cv2.MSER_create()
    # detector = cv2.FeatureDetector_create('MSER')
    fs = detector.detect(img)

    fs.sort(key=lambda x: -x.size)


    def supress(x):
        for f in fs:
            distx = f.pt[0] - x.pt[0]
            disty = f.pt[1] - x.pt[1]
            dist = math.sqrt(distx * distx + disty * disty)
            if (f.size > x.size) and (dist < f.size / 2):
                return True


    sfs = [x for x in fs if not supress(x)]

    for f in sfs:
        cv2.circle(img, (int(f.pt[0]), int(f.pt[1])), int(f.size / 2), d_red, 2, cv2.LINE_AA)
        cv2.circle(img, (int(f.pt[0]), int(f.pt[1])), int(f.size / 2), l_red, 1, cv2.LINE_AA)

    h, w = orig.shape[:2]
    vis = np.zeros((h, w * 2 + 5), np.uint8)
    print(vis.shape)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    print(vis.shape)
    vis[:h, :w] = orig
    vis[:h, w + 5:w * 2 + 5] = img

    cv2.imshow("image", vis)
    cv2.imwrite("c_o.jpg", vis)
    cv2.waitKey()
    cv2.destroyAllWindows()'''

    '''
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img=255-img # invert image
    # img = img[...,2]
    print(img.shape)
    img = cv2.GaussianBlur(img, (41, 41), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img)
    cv2.circle(img_original, maxLoc, 5, (255, 0, 0), 2)
    print(minVal, maxVal, minLoc, maxLoc)

    # img=cv2.blur(img, (10, 10))
    # img=cv2.GaussianBlur(img, (5, 5), 0)

    # (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # (thresh, img) = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY )
    # cv2.imshow('xxx', np.hstack((img, img_original[...,0])))
    cv2.imshow('xxx', img_original)
    cv2.waitKey(0)
    '''
    '''
    img = cv2.GaussianBlur(img, (41, 41), 0)
    cv2.imshow('xxx', img)
    # cv2.waitKey(0)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 150,
                               param1=50, param2=25 , minRadius=0, maxRadius=0)
    print(circles)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(img, (x, y), r, (250, 250, 250), 2)
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # show the output image
        cv2.imshow("output", img)
        cv2.waitKey(0)

    '''
