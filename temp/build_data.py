import multiprocessing
# from pandas import DataFrame
from .parameters import SEED, PATH_DATA, PATH_ORIGINAL, PATH_SAMPLE
from random import randint
from .tools import *

# Todo:
#   - Make it reproducing results
#   - saves less images(19.936 iso 20.000) Why?
# Global variables
# _sample_labels_df = DataFrame()

# Model Parameters
_im_size = 512
_balance = 3
_square_min_box = False
_n_samples = 1  # nr of samples per class
_workers = 1

"""
Changes:
v3:
    - Multiprocess image processing
V4:
    
"""


def process_images(i, fname, label, tot):
    img = LabelProcess.get_image_for_path(img_name=fname, path=PATH_ORIGINAL / 'train')

    if _balance == 3:
        ang = randint(0, 360)
        img = ImageAugmentation.rotate_image(img, angle=ang)
    else:
        ang = 'x'
    img = ImagePreprocess.auto_crop(img, square_min_box=_square_min_box)
    img = ImagePreprocess.resize(img, _im_size)
    # img = ImagePreprocess.make_edge_enhance(img)
    # img = ImagePreprocess.normalize_image(img)

    # raname jpeg to png lossless
    fname_png = '{}_{}.png'.format(fname.split('.')[0], ang)
    LabelProcess.save_image_to_path(img, name=fname_png, path=PATH_SAMPLE)
    print('{}/{} \t\t{}\t\t\t {}'.format(i, tot, fname_png, label))
    return fname_png, label


def engine():
    # Todo: backup and create new smpl_train
    # Todo: print start and end time

    sample_labels_df = DataFrame()
    labels_df: DataFrame = LabelProcess.get_labels()
    labels: list = labels_df['label'].unique()
    label_count = labels_df.groupby('label').agg('count')
    n_min = label_count['image'].min()

    if _balance == 1:  # each label gets the same number of images equal to that from the label with the least images
        for l in labels:
            sample_df = labels_df[labels_df['label'] == l]  # filter on label
            sample_df = LabelProcess.get_sample(labels_df=sample_df, n=n_min)  # take a sample sith size == min label
            sample_labels_df = sample_labels_df.append(sample_df)
    elif _balance == 2:  # each label gets the same number of images equal n_samples
        for l in labels:
            imgs_in_label = label_count['image'][l]
            sample_df = labels_df[labels_df['label'] == l]  # filter on label
            sample_df = LabelProcess.get_sample(labels_df=sample_df, n=min(_n_samples, imgs_in_label))  # take a sample with size == min(n_sampels, #images for label)
            sample_labels_df = sample_labels_df.append(sample_df)
    elif _balance == 3:  # augment
        for l in labels:
            sample_df = labels_df[labels_df['label'] == l]  # filter on label
            sample_df = LabelProcess.get_sample(labels_df=sample_df, n=_n_samples, replace=True)
            sample_labels_df = sample_labels_df.append(sample_df)
    elif _balance == 0:
        sample_labels_df = labels_df

    # Image processing - Multiprocess
    # build arguments
    tot = sample_labels_df.shape[0]  # total nr of images
    args = [(i, fname, sample_labels_df.iloc[i]['label'], tot) for i, fname in enumerate(sample_labels_df['image'])]  # not sorted list!
    # args = sorted(args, key=lambda x: x[0]) # to sort the args on 1st element of (i, fname,...) but doesn't work on multiprocess
    pool = multiprocessing.Pool(_workers)
    sample_labels = pool.starmap(process_images, args)  # contains tupels of (fname_png, label)
    sample_labels_df = DataFrame(sample_labels, columns=['image', 'label'])
    # save the labels
    sample_labels_df.to_csv(PATH_SAMPLE / 'labels.csv')


if __name__ == '__main__':
    engine()
