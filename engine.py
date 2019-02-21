# --------------------------------------------------------------------------------------------------------
# 2019/02/20
# retinopathy - engine.py
# md
# --------------------------------------------------------------------------------------------------------

"""
Engine for running experiments
"""
from pathlib import Path
from typing import Union, List

from sacred import Experiment
from sacred.observers import MongoObserver
from build_data import *

ex = Experiment('my test')
ex.observers.append(MongoObserver.create())


@ex.config
def configuration():
    """Parameters for experiments"""

    # Paths
    path: Path = Path('../data')  # path to data
    path_src: Path = path / 'original/train'  # path to source dataset
    path_dst: Path = path / ''  # path to destination
    path_lbl: Path = path / 'original/trainLabels.csv'  # path to source labels

    # Create dataset
    preprocess: List = ['rotate', 'crop', '']  # list of image preprocesses
    image_size: int = 512  # size of the square image


@ex.capture
def experiment(image_size, preprocess, path_src, path_dst, path_lbl):
    create_dataset(image_size=image_size,
                   preprocess=preprocess,
                   path_src=path_src,
                   path_dst=path_dst,
                   path_lbl=path_lbl)


@ex.automain
def run_engine():
    experiment()
