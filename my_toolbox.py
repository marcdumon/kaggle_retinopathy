# --------------------------------------------------------------------------------------------------------
# 2019/02/21
# my_toolbox.py
# md
# --------------------------------------------------------------------------------------------------------

# Todo:
#  - Make a package of all my personal "helper" scripts to import in different projects
import os
import time

from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Union
# from PIL import Image, ImageFilter
import numpy as np
import cv2 as cv


class MyLogTools:
    # Todo: implement other log levels and media
    """
    Collection of function to display or write logs to file of database
    Level:
        - 1 = OK, WARNING, ERROR
        - 2 = WARNING, ERROR
        - 3 = ERROR
        - 4 = INFO
        - 5 = DEBUG
        - 0  = ALL
    Medium:
        - 1   = console
        - 10  = file
        - 100 = database
    110 means logging to csv-file and database

    Implemented functions are:
        -
    """
    level = 0
    medium = 1

    @classmethod
    def log(cls, msg: str, level=level, medium=medium):
        if level == 0:  # Console
            if medium == 1:
                now = datetime.now()
                print('{}\t{}'.format(now, msg))
            if medium == 2:
                pass
            if medium == 3:
                pass
        if level == 5:  # Console
            if medium == 1:
                now = datetime.now()
                print('{}\t{}'.format(now, msg))


class MyOsTools:
    """
    Collection of functions that do OS file and directory operations like read, write, create, rename and check.

    Implemented functions are:
        - get_filenames(cls, path: Path, ext: str = '') -> List:
        - create_directory(cls, path: Path, name: str) -> Dict:
        - move_files(cls, fnames: List, path_src: Path, path_dst: Path) -> Dict:
        - check_dir_exists(cls, path: Path, name: str) -> Dict:
        - check_files_exist(cls, fnames: List, path: Path) -> Dict:

    """

    @classmethod
    def delete_directory(cls):
        pass

    @classmethod
    def rename_directory(cls):
        pass

    @classmethod
    def delete_files(cls):
        pass

    @classmethod
    def rename_files(cls, path: Path, old_names: List, new_names: List, workers: int = 1) -> Tuple[bool, str]:
        """
        Renames all the files in the old_names list to names in the new_names list.

        Args:
            path: Path to directory where the files to be renamed are located
            new_names: List of new names. Order is important!
            old_names: List of old names. Order is important!
            workers: Number of multiprocess cores

        Return:
            Returns a tuple (succ, msg)
        """
        succ = False
        msg = ''
        return succ, msg

    # --------------------------------------------------------------------------------------------------------
    @classmethod
    def check_files_exist(cls, fnames: List, path: Path) -> Dict:
        """
        Checks if all the files from the list fnames exists in directory path
            Args:
                fnames: List of filenames to be checkt
                path: Path to directory where the files from fnames should exist

            Return:
                 Returns a dict {success, message, fmissing, fexcess}
        """
        succ = True
        msg = ''

        filenames = MyOsTools.get_filenames(path)
        fmissing = list(set(fnames) - set(filenames))
        fexcess = list(set(filenames) - set(fnames))

        if not fexcess:
            succ = succ and True
            msg = 'OK: All {} \t files exist in directory: {}'.format(len(fnames), path)
            MyLogTools.log(msg)
        else:
            succ = succ and False
            msg = 'ERROR: These {} \t files are missing in directory {}: {}{}'.format(len(fmissing), path,
                                                                                      fmissing[:30],
                                                                                      '...' if len(
                                                                                          fmissing) > 30 else '')
            MyLogTools.log(msg)
        # Todo: code handeling of excess files, now only missing files
        return {'success': succ, 'message': msg, 'fmissing': fmissing, 'fexcess': fexcess}

    @classmethod
    def check_dir_exists(cls, path: str) -> Dict:
        """
        Checks if the directory exists

        Args:
            path: Path to the directory to check

        Returns:
            Returns a dict {success, message}
        """
        path_dir = os.fspath(path)  # Path -> PathLike
        succ = os.path.isdir(path_dir)
        if succ:
            msg = 'OK: Directory Exists: {}'.format(path_dir)
            MyLogTools.log(msg)
        else:
            msg = 'WARNING: Directory Doesn\'t Exist: {}'.format(path_dir)
            MyLogTools.log(msg)
        return {'success': succ, 'message': msg}

    @classmethod
    def move_files(cls, fnames: List, path_src: str, path_dst: str) -> Dict:
        """
        Moves all the files in the fnames list from path_src to path_dst. If the file already exists,
        it will be silently overwritten. If the file doesn't exist in the path_src, an error will be trown and
        the list with missing files will be returned

        Args:
            fnames: List of file names.
            path_src: Path to source directory
            path_dst: Path to destination directory

        Return:
            Returns a dict {succ, msg, missing_files}
        """
        succ = True
        errors = []
        msgs = []
        tot = len(fnames)
        for i, f in enumerate(fnames):
            try:
                os.rename(path_src + f, path_dst + f)
                succ = succ and True
                msg = 'OK: File {}/{} moved to : {}/{}'.format(i, tot, path_dst, f)
                MyLogTools.log(msg)
            except FileNotFoundError:
                succ = succ and False
                msg = 'ERROR: File {}/{} not found: {}'.format(i, tot, f)
                MyLogTools.log(msg)
                msgs.append(msg)
                errors.append(f)
        return {'success': succ, 'messages': msgs, 'errors': errors}

    @classmethod
    def get_filenames(cls, path: Path, ext: str = '') -> List:
        """
        Reads the filenames all the filenames in directory located in path or if ext is supplied,
        reads only thr filenames with that extension.

        Args:
            path: Path to the directory where the files are located
            ext: Extension to filter the returned filenames

        Returns:
            Returns a (alphabetically) sorted list of filenames
        """
        directory = os.scandir(path)
        fnames = [f.name for f in directory if f.is_file()]
        if ext:
            fnames = [f for f in fnames if f.split('.')[1] == ext]
        directory.close()
        fnames.sort()
        return fnames

    @classmethod
    def create_directory(cls, path: str) -> Dict:
        """
        Creates a new directory located in path. If the directory already exists, returns success flag = False
        If the directory already exists, a warning will be thrown and the list with existing directory will be returned

        Args:
            path: Path to location where the new directory should be created

        Returns:
            Returns a dict {success, message}

        """
        msgs = []
        path_dir = os.fspath(path)  # Path -> PathLike
        try:
            os.mkdir(path=path_dir)
            succ = True
            msg = 'OK: Directory Created: {}'.format(path_dir)
            MyLogTools.log(msg)
        except FileExistsError:
            succ = False
            msg = 'WARNING: Directory Exists: {}'.format(path_dir)
            MyLogTools.log(msg)
            msgs.append(msg)
        except FileNotFoundError:
            succ = False
            msg = 'ERROR: Path doesn\'t exists: {}'.format(path_dir)
            MyLogTools.log(msg)
            msgs.append(msg)
        return {'success': succ, 'message': msgs}


class MyImageTools:
    """
    Collection of functions to amnipulate images

    Implemented functions are:
        - get_image(cls, im_name: str, path: Path) -> Image:
        - auto_crop(cls, image: Image, square_min_box: bool = False) -> Image:
        - resize(cls, image: Image, size: int = 32, resample=Image.LANCZOS)->Image:

    """
    """ 
    Todo
    All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images 
    of shape (3 x H x W), where H and W are expected to be at least 224. 
    The images have to be loaded in to a range of [0, 1] and then 
    normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. 
    You can use the following transform to normalize:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    """

    @classmethod
    def get_image(cls, fname: str, path: str) -> np.array:
        """
        Gets image with name 'im_name' located in 'path' directory
            Args:
                - im_name: The filename of the image
                - path: The path where the image exists
            Returns:
                Returns an Image
        """
        im_array = cv.imread(path + fname, 1)
        if im_array is None:
            MyLogTools.log('ERROR: No image array loaded: {}{}'.format(path, fname))
            raise FileNotFoundError
        # MyLogTools.log('DEBUG: Image loaded: {}'.format(fname), level=5)
        return im_array

    @classmethod
    def save_image(cls, im_array: np.array, path: str, fname: str) -> None:
        cv.imwrite(path + fname, im_array)

    @classmethod
    def auto_crop(cls, im_array: np.array, square_min_box: bool = False) -> np.array:
        """
        Automatically crops an image and make it quare.
        Doing auto_crop before resizing avoids change in aspect ratio for rectangle images
            Args:
                im_array: The image to be cropped
                square_min_box: If True, the image will be croped with the min(width, hight) of the getbbox crop-box.
                                A part of the image be lost.
                                If False it wil take the max(with, height)
            Returns:
                Returnes the croped square image
        """
        # blured = im_array.filter(ImageFilter.BoxBlur(20))
        blured = cv.blur(im_array, (5, 5))
        # Convert to grayscale
        blured = cv.cvtColor(blured, cv.COLOR_BGR2GRAY)
        # Convert to B&W
        _, bw = cv.threshold(blured, 50, 255, cv.THRESH_BINARY)  # Eye circle is white, rest black
        # Get the bounding rectangle
        x, y, w, h = cv.boundingRect(bw)
        # Crop im_array to square matrix
        if square_min_box:
            d = min(w, h)
        else:
            d = max(w, h)
        im_array = im_array[y:y + d, x:x + d]
        return im_array

    @classmethod
    def resize(cls, im_array: np.array, size: int = 32, resample=cv.INTER_LINEAR) -> np.array:
        """
        Resizes an im_array to size size.
            Args:
                im_array: The im_array to be resised
                size: The size in px to wich the im_array shoul be resized
                resample: the im_array resampling algorithm
            Return:
                Returns the resized im_array
        """
        im_array = cv.resize(im_array, (size, size),
                             interpolation=cv.INTER_LINEAR)  # Todo: different interpolations give same reults
        return im_array

    @classmethod
    def minmax_scale(cls, im_array: np.array, per_channel: bool = False) -> np.array:
        """
        Scales a numpy image array per channel to [0, 1] and returns that array.
            Args:
                - im_array: numpy array to be min-max scaled per channel
                - per_channel: if True, then the min and max will be calculated per channel. Otherwise min and max will be calculated for the entire image
            Return:
                Returns the min-max-scaled numpy array of the image
        """
        # Todo: Per channel better ???
        # im_array.setflags(write=1)  # Todo: array seems read-only. Why?
        n_chanels = im_array.shape[-1]
        print(n_chanels)
        if per_channel:
            new_array = np.zeros(im_array.shape)
            for i in range(n_chanels):  # RGB
                m, M = im_array[..., i].min(), im_array[..., i].max()
                print(i, m, M)
                new_array[..., i] = (im_array[..., i] - m) / (M - m)
        else:
            new_array = (im_array - im_array.min()) / (im_array.max() - im_array.min())
        return new_array * 255

    @classmethod
    def standardize(cls, im_array: np.array, mean: List, std: List) -> np.array:
        # Todo: Test if this works correctly
        """
        See: https://github.com/tensorpack/tensorpack/issues/789
        Makes the features have 0-mean and unit (1) variance (or std). If the im_array is in BGR (OpenCV) iso RGB then the
        mean en std lists should also have that same order! If they are RGB means or std, swap color channels with
        mean[::-1] and std[::-1]
        Models are specific on the mean and std of the imput they are trained on! If you use a 3th party pretrained model,
        standardize the new im_array with the same mean and std as the pictures used to train the model on.
        Ex: Image Net used to train Resnet has mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

            Args:
                - im_array: the im_array to standardize. The im_array should be scaled to [0, 1]
                - mean: The mean  is a ist with value's for each color channel.
                - str: The standard deviation is a List with vaue's for each color schannel.
            Return:
                 Returns the standardized im_array
        """
        n_chanels = im_array.shape[-1]
        new_array = np.zeros(im_array.shape)
        for i in range(n_chanels):  # RGB
            new_array[..., i] = (im_array[..., i] - mean[i]) / std[i]
        return new_array

    @classmethod
    def ___calc_channel_mean_std(cls):
        # see: https://stackoverflow.com/questions/47850280/fastest-way-to-compute-image-dataset-channel-wise-mean-and-standard-deviation-in
        # https://stats.stackexchange.com/questions/374410/how-to-calculate-the-image-datasets-mean-and-std-for-deep-learning
        #

        """
        Calculates the per channel mean and std for a dataset.


        :return:
        """
        pass

    @classmethod
    def ___calc_mean_std_dataset(cls):
        # Multiprocess iterate over images
        # def get_mean_and_std(dataset): https://github.com/QuantScientist/Deep-Learning-Boot-Camp/blob/master/Kaggle-PyTorch/PyTorch-Ensembler/utils.py
        #
        pass


if __name__ == '__main__':
    xsdfsdf = 'yyy'
    # # x = MyOsTools.get_dir_filesnames(Path('../data/0_original/train'), 'jpeg')
    # # x = MyOsTools.create_directory(path=Path('../data/0_original/'), name='test2')
    # MyLogTools.log('xxx')
    # print(x)
    # print(len(x))
