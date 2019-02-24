# --------------------------------------------------------------------------------------------------------
# 2019/02/20
# retinopathy - build_data.py
# md
# --------------------------------------------------------------------------------------------------------

"""
Tools to build physical datasets and process labels
"""
from my_toolbox import MyOsTools as my_ot
from pathlib import Path
from typing import List
import pandas as pd

from engine import ex


def make_sample_labels(path_lbl: Path, balance: bool = True) -> pd.DataFrame:
    """
    Loads the original labels in a dataframe, takes a balanced or non-balanced sample of n_samples and
    returns the sample dataframe.

        Params:
            path_lbl: the full path (filename icluded) of the original label file.
                      The expected columns are ['image','level']

        Returns:
            Returns labels_dst_df, a datafrane containing a sample of the original labels.
            The columns are ['fname','ext','label'].
     """

    labels_src_df = pd.read_csv(path_lbl)
    labels_src_df['ext'] = 'jpeg'
    labels_src_df.columns = ['fname', 'label', 'ext']
    labels_src_df = labels_src_df.reindex(columns=['fname', 'ext', 'label'])  # swap column 'label' with 'ext'
    return labels_src_df


def create_destination_labels(labels_src_df: pd.DataFrame) -> pd.DataFrame:
    pass
    return labels_src_df


def create_dataset(image_size: int,
                   preprocess: List,
                   path_src: Path,
                   path_dst: Path,
                   path_lbl: Path) -> pd.DataFrame:
    labels_src_df = make_sample_labels(path_lbl)
    print(labels_src_df)

    labels_dst_df = pd.DataFrame(columns=['nr', 'fname', 'ext', 'label'])

    return labels_dst_df
