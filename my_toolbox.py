# --------------------------------------------------------------------------------------------------------
# 2019/02/21
# my_toolbox.py
# md
# --------------------------------------------------------------------------------------------------------

# Todo:
#  - Make a package of all my personal "helper" scripts to import in different projects
import os

from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Union
from PIL import Image, ImageFilter


class MyLogTools:
    # Todo: implement other log levels and media
    # Todo: make configuration variables global in other config file or script
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
        os.rename()

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
    def check_dir_exists(cls, path: Path, name: str) -> Dict:
        """
        Checks if the directory exists

        Args:
            path: Path to the directory to check
            name: Name of the directory to check

        Returns:
            Returns a dict {success, message}
        """
        path_dir = os.fspath(path / name)  # Path -> PathLike
        succ = os.path.isdir(path_dir)
        if succ:
            msg = 'OK: Directory Exists: {}'.format(path_dir)
            MyLogTools.log(msg)
        else:
            msg = 'WARNING: Directory Doesn\'t Exist: {}'.format(path_dir)
            MyLogTools.log(msg)
        return {'success': succ, 'message': msg}

    @classmethod
    def move_files(cls, fnames: List, path_src: Path, path_dst: Path) -> Dict:
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
                os.rename(path_src / f, path_dst / f)
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
    def create_directory(cls, path: Path, name: str) -> Dict:
        """
        Creates a new directory located in path. If the directory already exists, returns success flag = False
        If the directory already exists, a warning will be thrown and the list with existing directory will be returned

        Args:
            path: Path to location where the new directory should be created
            name: Name of the new directory to be created at path location

        Returns:
            Returns a dict {success, message}

        """
        msgs = []
        path_dir = os.fspath(path / name)  # Path -> PathLike
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

    @classmethod
    def get_image(cls, fname: str, path: Path) -> Image:
        """
        Gets image with name 'im_name' located in 'path' directory
            Args:
                - im_name: The filename of the image
                - path: The path where the image exists
            Returns:
                Returns an Image
        """
        img = Image.open(path / fname)
        # MyLogTools.log('DEBUG: Image loaded: {}'.format(fname), level=5)
        return img

    @classmethod
    def auto_crop(cls, image: Image, square_min_box: bool = False) -> Image:
        """
        Automatically crops an image and make it quare.
        Doing auto_crop before resizing avoids change in aspect ratio for rectangle images
            Args:
                image: The image to be cropped
                square_min_box: If True, the image will be croped with the min(width, hight) of the getbbox crop-box.
                                A part of the image be lost.
                                If False it wil take the max(with, height)
            Returns:
                Returnes the croped square image
        """
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
        # MyLogTools.log('OK: Image')
        return croped

    @classmethod
    def resize(cls, image: Image, size: int = 32, resample=Image.LANCZOS) -> Image:
        """
        Resizes an image to size size.
            Args:
                image: The image to be resised
                size: The size in px to wich the image shoul be resized
                resample: the image resampling algorithm
            Return:
                Returns the resized image
        """
        image = image.resize((size, size), resample=resample)
        return image


if __name__ == '__main__':
    x = 'yyy'
    # # x = MyOsTools.get_dir_filesnames(Path('../data/0_original/train'), 'jpeg')
    # # x = MyOsTools.create_directory(path=Path('../data/0_original/'), name='test2')
    # MyLogTools.log('xxx')
    # print(x)
    # print(len(x))
