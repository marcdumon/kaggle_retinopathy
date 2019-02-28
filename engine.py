# --------------------------------------------------------------------------------------------------------
# 2019/02/20
# retinopathy - engine.py
# md
# --------------------------------------------------------------------------------------------------------

"""
Engine for running experiments

x Todo: delete my_config
x Todo: implement image standardisation
x Todo: make test script that loads a single image to test different tools


Todo: move all pathlib paths to my_toolbox, use str as parameter.
Todo: change PIL for Opencv ?
Todo: implement calculate batch and dataset mean and std
                    (see:https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/3)
                    def get_mean_and_std(dataset): https://github.com/QuantScientist/Deep-Learning-Boot-Camp/blob/master/Kaggle-PyTorch/PyTorch-Ensembler/utils.py
Todo: implement ZCA whitening see
    - https://stackoverflow.com/questions/41635737/is-this-the-correct-way-of-whitening-an-image-in-python/41894317
Todo: add other preprocessing functions


Todo: load config.yalm into the sacred Experiment
Todo: check if numpy uses Atlas/Blas (performance)
"""
import multiprocessing
from pathlib import Path
from pprint import pprint
from typing import Union, List
from numpy.random import RandomState
from pandas import DataFrame, read_csv, concat
from sacred import Experiment
from sacred.observers import MongoObserver
from multiprocessing import Pool
# from build_data import create_dataset
from my_toolbox import MyOsTools as my_ot
from my_toolbox import MyLogTools as my_lt
from my_toolbox import MyImageTools as my_it
from ruamel_yaml import YAML  # as yaml

# from cv2 import * # conflict between python min and cv2 min !!!
import cv2


# ----------------------------------------------------------------------------------------------------------------------
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
    path_dset = Path('/mnt/Datasets/kaggle_diabetic_retinopathy/0_original/')
    fpath_lbl = path_dset / 'trainLabels.csv'

    # Standardize labels to: [fname, ext, label [, feature, ...]]
    labels_df = read_csv(fpath_lbl)
    labels_df['ext'] = 'jpeg'
    labels_df.columns = ['fname', 'label',
                         'ext']  # Todo: add fname+ext column and remove the df.str.cat()'s in the code. Change fname to iname
    labels_df = labels_df.reindex(columns=['fname', 'ext', 'label'])  # swap column 'label' with 'ext'
    # Add features Patientnr. and Left/Right Eyr
    feat_ds = labels_df.loc[:, 'fname'].str.split('_', expand=True)
    labels_df['f_patient'] = feat_ds.loc[:, 0]
    labels_df['f_eye'] = feat_ds.loc[:, 1]
    # Save labels.csv in train directory
    labels_df.to_csv(path_dset / 'train/labels.csv')
    # Reorder training dataset from /train/... to /train/label1/...
    # Create /train/label directories
    my_ot.create_directory(path_dset, 'train')  # parent
    labels_lst = labels_df['label'].unique()
    labels_lst.sort()
    for l in labels_lst:
        my_ot.create_directory(path_dset / 'train', str(l))  # children
    # Move the files to the correct new label-directories
    for l in labels_lst:
        path_src = path_dset / 'train'
        path_dst = path_dset / 'train' / str(l)
        fnames = ['{}.{}'.format(l[0], l[1]) for l in labels_df.loc[labels_df['label'] == l, ['fname', 'ext']].values]
        my_ot.move_files(fnames=fnames, path_src=path_src, path_dst=path_dst)


# ----------------------------------------------------------------------------------------------------------------------
yaml = YAML(typ='safe')
with open('config.yaml') as fl:
    config = yaml.load(fl)


# ex = Experiment(config['experiment_name'] + '_' + str(config['experiment_nr']))
# ex.observers.append(MongoObserver.create())


# ----------------------------------------------------------------------------------------------------------------------

class Engine:
    """

    """
    # Image_size ????

    # Todo: Is it not better to set all the class variables in a method?
    _path = Path(config['path'])
    _path_src_ds = _path / config['path_src_ds']
    _dataset_name = '_{}px_{}i_{}bt_{}'.format(config['image_size'], config['n_samples'], config['balance_type'],
                                               config['preprocess'])
    _dataset_name = _dataset_name.replace('[', '').replace(']', '').replace('\'', '').replace(',', 'X').replace(' ', '')
    _path_dst_ds = _path / (config['path_dst_ds'] + _dataset_name)
    _preprocess = config['preprocess']
    _n_samples = config['n_samples']
    _balance_type = config['balance_type']
    _random_state = config['seed']
    _workers = config['workers']
    _image_size = config['image_size']
    _save_ext = config['save_ext']
    _labels_df = read_csv(_path_src_ds / 'labels.csv',
                          index_col=[0])  # Dataframe with coulmns [fname, ext, label, f_...]
    _label_count = _labels_df.groupby(['label']).agg('count')['fname']  # Series with [label, count]
    _labels_uniq_lst = _labels_df['label'].unique().tolist()  # list of unique labels
    _labels_uniq_lst.sort()

    @classmethod
    # @ex.capture
    def create_dataset(cls):
        """
        Collection of operations to create a new dataset
        Return:
        """
        # Sanity checks
        # Check if source directory path_src_ds exists
        res = my_ot.check_dir_exists(path=cls._path_src_ds.parent, name=cls._path_src_ds.parts[-1])
        if not res['success']: raise SystemExit(0)
        # Check if labels.csv exists in .../train directory
        res = my_ot.check_files_exist(fnames=['labels.csv'], path=cls._path_src_ds)
        if not res['success']: raise SystemExit(0)
        # Check if dataset in path_src_ds contains all files and are in the right label directory
        for l in cls._labels_uniq_lst:
            fnames = cls._labels_df.loc[cls._labels_df['label'] == l, ['fname', 'ext']]
            fnames = fnames['fname'].str.cat(fnames['ext'], sep='.')
            res = my_ot.check_files_exist(fnames=fnames, path=cls._path_src_ds / str(l))
            if not res['success']: raise SystemExit(0)
        # Setup directory structure for _path_dst_ds
        cls._setup_dir_structure()
        # Take a sample from the dataset
        df = cls._take_samples()
        # Start preprocessing
        # combine fname and ext
        df['fname_save'] = df['fname'] + '.' + cls._save_ext
        df['fname_load'] = df['fname'].str.cat(df['ext'], sep='.')
        print(df.columns)

        args = [(f['fname_load'], f['fname_save'], f['label'], i + 1, df.shape[0]) for i, f in df.iterrows()]
        pool = multiprocessing.Pool(cls._workers)
        pool.starmap(cls._prepro_image, args)

        # Clean df for labels.csv
        df['ext'] = cls._save_ext
        df.to_csv(path_or_buf=cls._path_dst_ds / 'labels.csv')
        # Write the config to the dataset directory
        with open(cls._path_dst_ds / 'used_config.yaml', 'w') as f:
            yaml.dump(config, f)

    @classmethod
    # @ex.capture
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
                df_s = df_s.sample(n=ns, replace=False, random_state=cls._random_state)
            if cls._balance_type == 1:
                df_s = df_s.sample(n=cls._n_samples, replace=True, random_state=cls._random_state)
            if cls._balance_type == 2:
                ns = min(cls._n_samples, min(cls._label_count))
                df_s = df_s.sample(n=ns, replace=False, random_state=cls._random_state)
            if cls._balance_type == 3:
                # Todo: implement augmentation
                pass
            if cls._balance_type == 3:
                # Todo: implement pseudo labels
                pass
            df = concat([df, df_s])
        df = df.reset_index(drop=True)  # Old index now a column
        return df

    @classmethod
    # @ex.capture
    def _setup_dir_structure(cls):
        # Todo: doesn't exit when .../train_xxx_yyy/label1 is not empty
        """
        Setup the directory structure for the new dataset
        Directories will have following structure:
            - .../dataset/traing
            - .../dataset/taring/labels.csv
            - .../dataset/traing/label_1/
                        ...
            - .../dataset/traing/label_n/

            Return:
        """
        # Check if path_dst_ds exists, throw error when it's not empy, create if it doesn't exist
        res = my_ot.check_dir_exists(path=cls._path_dst_ds.parent, name=cls._path_dst_ds.parts[-1])
        if res['success']:  # _path_dst_ds exists
            res = my_ot.get_filenames(path=cls._path_dst_ds)  # check if _path_dst_ds is empty
            if res:  # _path_dst_ds is not empty
                my_lt.log('ERROR: Directory not empty: {}'.format(cls._path_dst_ds))
                raise SystemExit(0)
        else:  # create _path_dst_ds directory
            res = my_ot.create_directory(path=cls._path_dst_ds.parent, name=cls._path_dst_ds.parts[-1])
            if not res['success']: raise SystemExit(0)
        # Make label directories
        for l in cls._labels_uniq_lst:
            res = my_ot.check_dir_exists(path=cls._path_dst_ds, name=str(l))
            if res['success']:  # _path_dst_ds/label/ exists
                res = my_ot.get_filenames(path=cls._path_dst_ds / str(l))  # check if _path_dst_ds/label/ is empty
                if res:  # _path_dst_ds/label/ is not empty
                    my_lt.log('ERROR: Directory not empty: {}'.format(cls._path_dst_ds / str(l)))
                    raise SystemExit(0)
            else:  # create _path_dst_ds directory
                res = my_ot.create_directory(path=cls._path_dst_ds, name=str(l))
                if not res['success']: raise SystemExit(0)

    @classmethod
    # @ex.capture
    def _prepro_image(cls, fname_load: str, fname_save: str, label: Union[int, str], i: int = 0, tot: int = 0) -> None:
        """
        Collection of operations to preprocess an image. It loads an image and executes preprocessing operations
        according to the values of prepro.
        It returns ???
            Args:
                - path: The path where the image is located
                - fname: The name and extension of the imagefile
                - prepro_lst: List containing preprocess operations to be executed in sequence
                - i: image counter
                - tot: totals amout of images
            Returns:
                ???
        """
        # Load image
        img = my_it.get_image(fname=fname_load, path=cls._path_src_ds / str(label))
        # Start preprocessing
        for p in cls._preprocess:
            if p == 'autocrop':
                img = my_it.auto_crop(image=img)
            if p == 'resize':
                img = my_it.resize(image=img, size=cls._image_size)
            if p == 'scale01':
                img = my_it.scale_01(img)
            if p == 'stdardize':
                my_it.standardize(img, mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])  # This sould be in config.yaml

        # save image
        img.save(cls._path_dst_ds / str(label) / fname_save)
        my_lt.log('INFO: Image {}/{} \t\t saved: {}'.format(i, tot, fname_save))
        return img


def make_config_yaml():
    # Todo: To have random, do: if seed==0, seed=randint() ?
    """
        Process that thakes all the default parameters from a string and saves it als a config.yalm file.

        Return: None
    """

    # !!!! The intendations in doc must be alligned to the far left otherwise the yaml file looks ugly
    doc = """
# Configuration File

# Setup
do_setup: False # If True, edit the setup() code for each new dataset
do_make_config: True # If True, overwrite the config.yaml file with default one.
workers: 16     # nr of cores used in multiprocess. Max=16
seed: 42        # Seed for replication
experiment_name: test
experiment_nr: 0

# Image settings
image_size: 512  # size of the square image
n_samples: 3  # nr of images per label
# balance_type; how to balance the labels.
    # 0: take n_samples. if not enough, continue. Can lead to imbalanced labels
    # 1: take n_samples. if not enough, duplicate images till each label has n_samples images
    # 2: sample all labels for an number of images equal to the amount of images in the lowest label
    # 3: take n_samples. if not enough, augment images till each label has n_samples images
balance_type: 1
# png is lossless
save_ext: 'png' # The extension dictates the compression algorithm 
# list of image preprocesses
    # autocrop
    # resize
    # scale01
    # stdardize
preprocess:     
    - autocrop
    - resize
    - scale01

# Paths
path: ../data                   # path to dataset for the project. 
                                # Usualy it's to a linked directory to the fast ssd drive
path_src_ds: 0_original/train   # path to source dataset
path_dst_ds: experiments/train  # path to destination
    """
    yaml = YAML(typ='rt')
    yaml.indent(mapping=2, sequence=4, offset=4)
    yaml_doc = yaml.load(doc)
    with open('config.yaml', 'w') as f:
        yaml.dump(yaml_doc, f)


# @ex.main
def experiment():
    if config['do_setup']: setup()
    if config['do_make_config']: make_config_yaml()
    Engine.create_dataset()


# @ex.automain
# def run_engine():
#     experiment()


if __name__ == '__main__':
    experiment()
