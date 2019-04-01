# --------------------------------------------------------------------------------------------------------
# 2019/02/20
# retinopathy - engine_dataset.py
# md
# --------------------------------------------------------------------------------------------------------

"""


!!! Todo: check out profesional code from http://albumentations.readthedocs.io how to make my software more professional

Todo: Better crop: inside the eye-ball iso outside. The assumption is that the relevant info lays inside and the data on the border of the eye can be discarted



Todo: test if create_train_dataset still works
Todo: merge create_train_dataset and create_test_dataset if possible

Todo: improve the random.seed in different modules. Doesn't seem to work in multiprocess. I get different angle each time in rotation
Todo: change all type hints from np.array to np.ndarray
Todo: To have random, do: if seed==0, seed=randint() ?
Todo: implement other log levels and media
Todo: Test if def setup(): still works after changing Path to str

Todo: check if numpy uses Atlas/Blas (performance)

Todo: implement preprocess test dataset and prepare submissions
Todo: implement baseline classifier
Todo: implement data augmentation
Todo: implement feature extraction (sift)
Todo: add other preprocessing functions (rotate, flip, ...)
Todo: implement calculate batch and dataset mean and std
                    (see:https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/3)
                    def get_mean_and_std(dataset): https://github.com/QuantScientist/Deep-Learning-Boot-Camp/blob/master/Kaggle-PyTorch/PyTorch-Ensembler/utils.py
Todo: implement ZCA whitening see
    - https://stackoverflow.com/questions/41635737/is-this-the-correct-way-of-whitening-an-image-in-python/41894317
Todo: load config.yalm into the sacred Experiment
Todo: implement pseud-labeling
Todo: better error trapping like:
        class UnknownDatasetError(Exception):
            def __str__(self):
                return "unknown datasets error"

"""
import multiprocessing
import random
import time
import traceback
from typing import Union, Optional

from pandas import DataFrame, read_csv, concat
from ruamel_yaml import YAML  # as yaml

from settings import *
from my_toolbox import MyImageTools as my_it
from my_toolbox import MyLogTools as my_lt
from my_toolbox import MyOsTools as my_ot


def setup():
    """
    Changes for each different dataset.

    Collection of preliminary operations to "standarize" the original dataset.
    Doeing these opperations solves the problem of having dataset-specific code in
    the rest of the application. It makes the application more generic.
    generic.
        Minimum directory requirements:
            - .../dataset/traing
            - .../dataset/taring/labels.csv
            - .../dataset/traing/label_1/file_1
                        ...
            - .../dataset/traing/label_n/file_m
        Minimum labels requirement
            - minimum columns: fname, ext, label
            - optional feature columns: f_xxx
    """
    path_dset = CONFIG['path_src_ds']
    fpath_lbl = path_dset + 'trainLabels.csv'
    labels_df = read_csv(fpath_lbl)
    labels_df['ext'] = 'jpeg'
    labels_df.columns = ['fname', 'label', 'ext']
    labels_df = labels_df.reindex(columns=['fname', 'ext', 'label'])  # swap column 'label' with 'ext'
    # Add features Patientnr. and Left/Right Eyr
    feat_ds = labels_df.loc[:, 'fname'].str.split('_', expand=True)
    labels_df['f_patient'] = feat_ds.loc[:, 0]
    labels_df['f_eye'] = feat_ds.loc[:, 1]
    # Save labels.csv in train directory
    labels_df.to_csv(path_dset + 'labels.csv')

    # Reorder training dataset from /train/... to /train/label1/...
    # Create /train/label directories
    my_ot.create_directory(path_dset + 'train')  # parent
    labels_lst = labels_df['label'].unique()
    labels_lst.sort()

    for l in labels_lst:
        my_ot.create_directory(path_dset + 'train/' + str(l))

    # Move the files to the correct new label-directories
    for l in labels_lst:
        path_src = path_dset + 'train/'
        path_dst = path_dset + 'train/' + str(l)
        fnames = ['{}.{}'.format(l[0], l[1]) for l in labels_df.loc[labels_df['label'] == l, ['fname', 'ext']].values]
        my_ot.move_files(fnames=fnames, path_src=path_src, path_dst=path_dst)


# ----------------------------------------------------------------------------------------------------------------------
class DatasetEngine:
    """
    DatasetEngine loads config.yalm parameters, and
    """
    error_images = []

    _preprocess = CONFIG['preprocess']
    if _preprocess is None: _preprocess = []

    if 'resize' not in _preprocess: CONFIG['image_size'] = 999999  # no resizing of pictures
    _dataset_name = '{}px_{}i_{}bt_{}/'.format(CONFIG['image_size'], CONFIG['n_samples'], CONFIG['balance_type'], CONFIG['preprocess'])
    _dataset_name = _dataset_name.replace('[', '').replace(']', '').replace('\'', '').replace(',', 'X').replace(' ', '')

    _path = CONFIG['path']
    _path_src_ds = CONFIG['path_src_ds']
    _path_dst_ds = CONFIG['path_dst_ds'] + _dataset_name
    print('-' * 120)
    print(_path_src_ds, _path_dst_ds, _path)
    print('-' * 120)

    _n_samples = CONFIG['n_samples']
    _balance_type = CONFIG['balance_type']
    _workers = CONFIG['workers']
    _seed = CONFIG['seed']
    _image_size = CONFIG['image_size']
    _load_ext = CONFIG['load_ext']
    _save_ext = CONFIG['save_ext']

    _mean = CONFIG['mean']
    _std = CONFIG['std']
    print(_mean, _std)
    _labels_df = read_csv(_path_src_ds + 'labels.csv', index_col=[0])  # Dataframe [fname, ext, label, f_...]
    _label_count = _labels_df.groupby(['label']).agg('count')['fname']  # Series with [label, count]
    _labels_uniq_lst = _labels_df['label'].unique().tolist()  # list of unique labels
    _labels_uniq_lst.sort()

    @classmethod
    def create_test_dataset(cls):
        """


        """
        # Create df for test dataset
        df = DataFrame(my_ot.get_filenames(cls._path_src_ds + 'test/', ext=cls._load_ext), columns=['fname_load'])
        df[['fname', 'ext']] = df['fname_load'].str.split('.', expand=True)  # +cls._save_ext

        df['fname_save'] = df['fname'] + '.' + cls._save_ext
        df['label'] = [-1] * df.shape[0]
        print(df.columns)
        df = df.drop('ext', axis=1)

        # # Start preprocessing
        # args = [(row['fname_load'], cls._path_src_ds + 'test/', row['fname_save'], cls._path_dst_ds + 'test/', '', False, i + 1, df.shape[0])
        #       for i, row in df.iterrows()]
        # # args = [(row['fname_load'], i + 1, df.shape[0]) for i, row in df.iterrows()]
        # start = time.time()

        # Start preprocessing
        args = [(row['fname_load'], cls._path_src_ds + 'test/', row['fname_save'], cls._path_dst_ds + 'test/', '', False, i + 1, df.shape[0], row)
                for i, row in df.iterrows()]  # Todo: we pass row (for writing to labels.csv) and row['...'] => redundant info
        start = time.time()

        with multiprocessing.Pool(cls._workers) as pool:
            res = pool.starmap(cls._prepro_image, args)  # All fnames from images with errors in _prepro_image are accumulated in res
        end = time.time()

        # save errors
        errors_df = df[df['fname_load'].isin(res)]
        print('Errors:')
        print(errors_df)
        errors_df.to_csv(path_or_buf=cls._path_dst_ds + 'error_test_images.csv')

        # save config to the dataset directory
        with open(cls._path_dst_ds + 'used_test_config.yaml', 'w') as f:
            yaml.dump(CONFIG, f)

        print('-' * 120)
        my_lt.log('INFO: Processing images took {} seconds or {} minutes '.format(end - start, int((end - start) / 60)))
        print('-' * 120)

    @classmethod
    def create_train_dataset(cls):
        """
        Collection of operations to create the test dataset
        Return:
        """
        # Sanity checks
        # Check if source directory path_src_ds exists
        res = my_ot.check_dir_exists(path=cls._path_src_ds)
        if not res['success']: raise SystemExit(0)

        # Check if labels.csv exists in _path_src_ds directory
        res = my_ot.check_files_exist(fnames=['labels.csv'], path=cls._path_src_ds)
        if not res['success']: raise SystemExit(0)

        # Check if dataset in path_src_ds/train contains all files and are in the right label directory
        for l in cls._labels_uniq_lst:
            fnames = cls._labels_df.loc[cls._labels_df['label'] == l, ['fname', 'ext']]
            fnames = fnames['fname'].str.cat(fnames['ext'], sep='.')
            res = my_ot.check_files_exist(fnames=fnames, path=cls._path_src_ds + 'train/' + str(l))
            if not res['success']: raise SystemExit(0)

        # Setup directory structure in _path_dst_ds
        cls._setup_dir_structure()

        # Create labels.csv
        with open(cls._path_dst_ds + 'labels.csv', 'w') as f:
            f.write('nr,fname,label,f_patient,f_eye,fname_save,fname_load\n')

        # Take a sample from the dataset
        df = cls._take_samples()

        # combine fname and ext
        df['fname_save'] = df['fname'] + '.' + cls._save_ext
        df['fname_load'] = df['fname'].str.cat(df['ext'], sep='.')
        df = df.drop('ext', axis=1)

        print(df.columns)
        print(df.groupby('label').agg('count').loc[:, 'fname'])

        # Start preprocessing
        args = [(row['fname_load'], cls._path_src_ds + 'train/', row['fname_save'], cls._path_dst_ds + 'train/', row['label'], True, i + 1, df.shape[0], row)
                for i, row in df.iterrows()]  # Todo: we pass row (for writing to labels.csv) and row['...'] => redundant info
        start = time.time()

        with multiprocessing.Pool(cls._workers) as pool:
            res = pool.starmap(cls._prepro_image, args)  # All fnames from images with errors in _prepro_image are accumulated in res
        end = time.time()

        # save errors
        errors_df = df[df['fname_load'].isin(res)]
        print('Errors:')
        print(errors_df)
        errors_df.to_csv(path_or_buf=cls._path_dst_ds + 'error_images.csv')

        # save config to the dataset directory
        with open(cls._path_dst_ds + 'used_config.yaml', 'w') as f:
            yaml.dump(CONFIG, f)

        print('-' * 120)
        my_lt.log('INFO: Processing images took {} seconds or {} minutes '.format(end - start, int((end - start) / 60)))
        print('-' * 120)

    @classmethod
    def _prepro_image(cls, fname_load: str, path_load: str, fname_save: str, path_save: str, label: Union[int, str],
                      make_flat_links: bool = False, i: int = 0, tot: int = 0, row=None) -> Optional[str]:
        """
        Collection of operations to preprocess an image. It loads an image and executes preprocessing operations
        according to the values of prepro.
        Be carefull what you return because it will accumulate in multiprocess pool and fill up memory !!!
            Args:
                - fname_load:
                - path_load:
                - fname_save:
                - path_save
                - make_flat_links: if true, put a link to the training images in directory 'train_flat/'
                - i: image counter
                - tot: totals amout of images
            Returns:
                Returns None.
        """

        # Load image
        im_array = my_it.get_image(iname=fname_load, path=path_load + str(label) + '/')
        # Start preprocessing
        try:
            for p in cls._preprocess:
                if p == 'augm':
                    aug = my_it.augment(im_array=im_array, rnd=True)
                    im_array = aug['im_array']
                    fname_ext = fname_save.split(sep='.')
                    fname_save = fname_ext[0] + '_' + aug['aug_name'] + '.' + fname_ext[1]
                if p == 'autocrop':
                    im_array = my_it.autocrop(im_array=im_array, inside=False)
                if p == 'autocrop_in':
                    im_array = my_it.autocrop(im_array=im_array, inside=True, expand=1.0)
                if p == 'resize':
                    im_array = my_it.resize(im_array=im_array, size=cls._image_size)
                if p == 'gray':
                    im_array = my_it.gray(im_array)
                if p == 'minmax':
                    im_array = my_it.minmax(im_array)
                if p == 'stdize':
                    my_it.stdize(im_array, mean=cls._mean, std=cls._std)
                if p == 'sift':
                    im_array = my_it.sift(im_array)
                if p == 'hist':
                    im_array = my_it.histogram_eqalization(im_array)
                if p == 'pca':
                    im_array = my_it.random_pca(im_array)

            # save image
            my_it.save_image(im_array=im_array, path=path_save + str(label) + '/', iname=fname_save)
            my_lt.log('INFO: Image {}/{} \t\t saved: {}'.format(i, tot, cls._path_dst_ds + str(label) + '/' + fname_save))  # todo: add 'train' or 'test' to the path

            if make_flat_links:
                # create symlink from .../train/im00.jpg to .../train/label/im00.jpg
                my_it.symlink_image(path_src=cls._path_dst_ds + 'train/' + str(label) + '/',
                                    path_dst=cls._path_dst_ds + 'train_flat/', iname=fname_save)

            if path_save.split('/')[-2] == 'train':  # if train
                # append row to the labels.csv
                row = str(i) + ',' + row['fname'] + ',' + str(row['label']) + ',' + str(row['f_patient']) + ',' + row['f_eye'] + ',' + fname_save + ',' + row['fname_load'] + '\n'
                with open(cls._path_dst_ds + 'labels.csv', 'a') as f:
                    f.write(row)
                    pass
                return None  # !!!! Don't return im_array because it will accumulate in multiprocess pool and fill up memeory

        except Exception as e:
            print('-' * 120)
            print("type error: " + str(e))
            print(traceback.format_exc())
            print('-' * 120)
            return fname_load  # The error fnames will be accumulated in the multiprocess pool

    @classmethod
    def _take_samples(cls):
        """
        Takes a sample of the _labels_df. Depending on the value of _balance_type,. the sample will be taken as:
                - 0: take n_samples. if not enough, continue. Can lead to imbalanced labels
                - 1: take n_samples. if not enough, duplicate images till each label has n_samples images
                - 2: sample all labels for an number of images equal to the amount of images in the smallest label
                - 3: take n_samples. if not enough, augment images till each label has n_samples images
                - 4: take n_samples. if not enough, calculate pseudolabels on the testset and add the images to the right label directory
            Return:
                Returns a dataframe with samples from _labels_df
        """
        df = DataFrame()
        for l in cls._labels_uniq_lst:
            df_s = cls._labels_df.loc[cls._labels_df['label'] == l, :]
            lc = cls._label_count[l]

            if cls._balance_type == 0:
                ns = min(cls._n_samples, lc)
                df_s = df_s.sample(n=ns, replace=False, random_state=cls._seed)  # Todo: is this necessary if np.random.seed is set in settings.np
            if cls._balance_type == 1:
                df_s = df_s.sample(n=cls._n_samples, replace=True, random_state=cls._seed)
            if cls._balance_type == 2:
                ns = min(cls._n_samples, min(cls._label_count))
                df_s = df_s.sample(n=ns, replace=False, random_state=cls._seed)
            if cls._balance_type == 3:
                ns = lc
                df_s = df_s.sample(n=cls._n_samples, replace=True, random_state=cls._seed)
            if cls._balance_type == 4:
                pass
            df = concat([df, df_s])
        df = df.reset_index(drop=True)  # Old index now a column
        return df

    @classmethod
    def _setup_dir_structure(cls, overwrite: bool = False):
        """
        Setup the directory structure for the new dataset
        Directories will have the resnet structure:
            - .../dataset/train
            - .../dataset/tarin/labels.csv
            - .../dataset/train/label_1/
                        ...
            - .../dataset/train/label_n/
            - .../dataset/valid/label_1/
                        ...
            - .../dataset/valid/label_n/
            - .../dataset/test/
            Return:
        """
        # Check if path_dst_ds exists, throw error when it's not empy, create if it doesn't exist
        res = my_ot.check_dir_exists(path=cls._path_dst_ds)
        if res['success']:  # _path_dst_ds exists
            # check if _path_dst_ds is empty
            res = my_ot.get_filenames(path=cls._path_dst_ds)
            if res and not overwrite:  # _path_dst_ds is not empty
                my_lt.log('ERROR: Directory not empty: {}'.format(cls._path_dst_ds))
                raise SystemExit(0)
        else:  # _path_dst_ds doesn't exist
            # create _path_dst_ds
            res = my_ot.create_directory(path=cls._path_dst_ds)
            if not res['success']: raise SystemExit(0)

            # create _path_dst_ds/train and _path_dst_ds/valid
            for p in [cls._path_dst_ds + 'train/', cls._path_dst_ds + 'valid/']:
                res = my_ot.create_directory(path=p)
                if not res['success']: raise SystemExit(0)
                # Make label directories
                for l in cls._labels_uniq_lst:
                    res = my_ot.create_directory(path=p + str(l))
                    if not res['success']: raise SystemExit(0)

            # create _path_dst_ds/test and _path_dst_ds/train_flat
            res = my_ot.create_directory(path=cls._path_dst_ds + 'test/')
            if not res['success']: raise SystemExit(0)
            res = my_ot.create_directory(path=cls._path_dst_ds + 'train_flat/')
            if not res['success']: raise SystemExit(0)


# ----------------------------------------------------------------------------------------------------------------------
def make_config_yaml():
    """
        Process that thakes all the default parameters from a string and saves it als a config.yalm file.

        Return: None
    """

    # !!!! The intendations in doc must be alligned to the far left otherwise the yaml file looks ugly
    doc = """
# Configuration File

experiment_name: test
experiment_nr: 0

# Setup
do_setup: False         # If True, edit the setup() code for each new dataset
do_make_config: False   # If True, overwrite the config.yaml file with default one.
workers: 16             # nr of cores used in multiprocess. Max=16
seed: 42                # Seed for replication

# The mean and std for using pretrained weights
# Ex: Resnet: mean:[0.485, 0.456, 0.406] and std: [0.229, 0.224, 0.225]
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]

# Image settings
image_size: 512  # size of the square image
n_samples: 3  # nr of images per label
# balance_type; how to balance the labels.
    # 0: take n_samples. if not enough, continue. Can lead to imbalanced labels
    # 1: take n_samples. if not enough, duplicate images till each label has n_samples images
    # 2: sample all labels for an number of images equal to the amount of images in the smallest label
    # 3: take n_samples. if not enough, augment images till each label has n_samples images
balance_type: 1

load_ext: 'jpeg'
# png is lossless
save_ext: 'png'         # The extension dictates the compression algorithm

# list of image preprocesses
    # augm # augement dataset. Must be first
    # hist
    # autocrop
    # autocrop_in       # autocrop but with square inside the eye-circle to reduce size
    # resize
    # minmax            # scales array to [0, 1]
    # stdize            # centers a minmax array around 0 with unit variance
    # gray
    # sift
    # pca
preprocess:
    # - augm
    # - hist
    # - autocrop
    # - resize
    # - gray
    # - minmax
    # - sift
    # - pca

# Paths
path: /mnt/Datasets/kaggle_diabetic_retinopathy/                     # path to dataset for the project.
path_src_ds: /mnt/Datasets/kaggle_diabetic_retinopathy/0_original/   # path to source dataset
path_dst_ds: /mnt/Datasets/kaggle_diabetic_retinopathy/experiments/  # path to destination

    """
    yaml = YAML(typ='rt')
    yaml.indent(mapping=2, sequence=4, offset=4)
    yaml_doc = yaml.load(doc)
    with open('config.yaml', 'w') as f:
        yaml.dump(yaml_doc, f)


# ----------------------------------------------------------------------------------------------------------------------
def experiment():
    if CONFIG['do_setup']: setup()
    if CONFIG['do_make_config']: make_config_yaml()
    DatasetEngine.create_train_dataset()
    # DatasetEngine.create_test_dataset()


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    experiment()
