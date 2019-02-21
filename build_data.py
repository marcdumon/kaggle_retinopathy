# --------------------------------------------------------------------------------------------------------
# 2019/02/20
# retinopathy - build_data.py
# md
# --------------------------------------------------------------------------------------------------------

"""
Tools to build physical datasets and process labels
"""
from pathlib import Path
from typing import List
import pandas as pd

from engine import ex


def create_directory(path):
    pass


def get_source_labels(path_lbl: Path) -> pd.DataFrame:
    """Loads the source labels in a dataframe and returns a cleaned [fname, ext, label] dataframe"""
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
    labels_src_df = get_source_labels(path_lbl)
    print(labels_src_df)

    labels_dst_df = pd.DataFrame(columns=['nr', 'fname', 'ext', 'label'])

    return labels_dst_df
