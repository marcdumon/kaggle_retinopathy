from .parameters import SEED, PATH_DATA, PATH_SAMPLE, PATH_ORIGINAL
from pathlib import Path
from typing import Any, Union
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from pandas import DataFrame

# Parameters # Todo: no need to convert global to local
_SEED = SEED
# _PATH_DATA = Path('../data')  # correct path from notebools ./nbs
# _PATH_ORIGINAL = _PATH_DATA / 'original'
# _PATH_SAMPLE = _PATH_DATA / 'smpl_train'
_PATH_DATA = PATH_DATA
_PATH_SAMPLE = PATH_SAMPLE
_PATH_ORIGINAL = PATH_ORIGINAL

"""
Changes:
- changed Image.BICUBIC to Image.LANCZOS
- min(box ... destroys information. -> implement max(box ... 
"""


def _make_path(path: Union[Path, str], suffix: str) -> Path:
    """
    Add suffix to path. If path is a file with extension, the suffix is added before extention
    """
    if type(path) is str:
        path = Path(path)
    parent = path.parents[0]
    # filename = path.name # =name[.ext]
    name = path.name.split('.')[0]
    if name == path.name:
        ext = ''
    else:
        ext = '.' + path.name.split('.')[1]
    nw_path = Path('{}/{}_{}{}'.format(parent, name, suffix, ext))
    return nw_path


class LabelProcess:
    def __index__(self):
        pass

    @classmethod
    def get_labels(cls, path: Path = _PATH_ORIGINAL,
                   name: str = 'trainLabels.csv') -> DataFrame:
        """Load the all labels in df"""
        _df = pd.read_csv(path / name)
        _df['image'] = _df['image'] + '.jpeg'
        _df.columns = ['image', 'label']
        return _df

    @classmethod
    def filter_labels(cls, labels_df: DataFrame, label: int) -> DataFrame:
        """Filter labels_df for label"""
        return labels_df[labels_df['label'] == label]

    @classmethod
    def get_sample(cls, labels_df: DataFrame, n: int = 0, replace: bool = True) -> DataFrame:
        return labels_df.sample(n=n, random_state=_SEED, replace=replace)
        pass

    @classmethod
    def get_image_for_path(cls, img_name: str,
                           path: Path = _PATH_ORIGINAL / 'train') -> Image:
        """ Gets image with name 'img_name' located in 'path' directory"""
        img = Image.open(path / img_name)
        return img

    @classmethod
    def save_image_to_path(cls, image: Image, name: str, path: Path = _PATH_SAMPLE,
                           label: int = None) -> str:
        image.save(path / name)
        if label:
            pass
        return name


class ImageAugmentation:
    pass

    @classmethod
    def rotate_image(cls, image: Image, angle: int = 0, expand: bool = True) -> Image:
        image = image.rotate(angle, resample=Image.BICUBIC, expand=expand)  # Todo: BICUBIC. LANCZOS is not implemented for rotate
        return image


class ImagePreprocess:
    """
        - crop image
        - gray
        - channels
        - colors
    """

    @classmethod
    def normalize_image(cls, image: Image) -> Image:
        """
        See: https://www.kaggle.com/gauss256/preprocess-images
        Normalize PIL image
        Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
        """
        img_y, img_b, img_r = image.convert('YCbCr').split()
        img_y_np = np.asarray(img_y).astype(float)
        img_y_np /= 255
        img_y_np -= img_y_np.mean()
        img_y_np /= img_y_np.std()
        scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                        np.abs(np.percentile(img_y_np, 99.0))])
        img_y_np = img_y_np / scale
        img_y_np = np.clip(img_y_np, -1.0, 1.0)
        img_y_np = (img_y_np + 1.0) / 2.0
        img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)
        img_y = Image.fromarray(img_y_np)
        img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))
        img_nrm = img_ybr.convert('RGB')
        return img_nrm

    @classmethod
    def resize(cls, image: Image, size: int = 2048) -> Image:
        image = image.resize((size, size), resample=Image.LANCZOS)
        return image

    @classmethod
    def make_equalize(cls, image: Image, mask=None) -> Image:
        image = ImageOps.equalize(image, mask=mask)
        return image

    @classmethod
    def auto_crop(cls, image: Image, square_min_box: bool = False) -> Image:
        blured = image.filter(ImageFilter.BoxBlur(20))
        bw = blured.convert('1')  # Convert to B&W for better box
        box = bw.getbbox()  # box is not a square
        # make min or max square box
        if square_min_box:  # This destroys information
            l = min(box[2] - box[0], box[3] - box[1])
        else:  # max
            l = max(box[2] - box[0], box[3] - box[1])
        center = ((box[2] + box[0]) / 2, (box[3] + box[1]) / 2)
        box = (center[0] - l / 2, center[1] - l / 2, center[0] + l / 2, center[1] + l / 2)
        croped = image.crop(box)
        return croped

    @classmethod
    def make_contour(cls, image: Image) -> Image:
        image = image.filter(ImageFilter.CONTOUR())
        return image

    @classmethod
    def make_edge_enhance(cls, image: Image) -> Image:
        image = image.filter(ImageFilter.EDGE_ENHANCE())
        return image

    @classmethod
    def make_edges(cls, image: Image) -> Image:
        image = image.filter(ImageFilter.FIND_EDGES())
        return image

    @classmethod
    def make_gray(cls, labels_df: DataFrame, read_path: Path = _PATH_SAMPLE):
        pass


if __name__ == '__main__':
    # print(_PATH_ORIGINAL)
    # df = LabelProcess.get_labels(False)
    # df = LabelProcess.filter_labels(df, 3)
    # df = LabelProcess.get_sample(labels_df=df, n=100)
    # print(df.shape)
    # print(df.head())
    LabelProcess.get_image_for_path('10_left.jpeg')
    # ImagePreprocess.make_gray()
