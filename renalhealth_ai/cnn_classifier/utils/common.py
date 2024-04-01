import os
from box.exceptions import BoxValueError
import yaml
from cnn_classifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """read yaml file and returns config

    :param path_to_yaml: path like input
    :type path_to_yaml: Path
    :raises:
       ValueError: if yaml file is empty
       e: empty file
    :return: config
    :rtype: ConfigBox
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    :param path_to_directories: list of path of directories to create
    :type path_to_directories: list
    :param verbose: show logs if True, defaults to True
    :type verbose: bool, optional
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")
        

@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    :param path: path to json file
    :type path: Path
    :param data: data to be saved in json file
    :type data: dict
    """
    with open(path, "w") as outfile:
        json.dump(data, outfile, indent=4)
        logger.info(f"json file: {path} saved successfully")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json data

    :param path: path to json file
    :type path: Path
    :return: data as class attributes instead of dict
    :rtype: ConfigBox
    """
    with open(path, "r") as json_file:
        data = json.load(json_file)
        logger.info(f"json file loaded successfully form: {path}")
        return ConfigBox(data)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    :param data: data to be saved as binary
    :type data: Any
    :param path: path to binary file 
    :type path: Path
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file: {path} saved successfully")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary file

    :param path: path to binary file
    :type path: Path
    :return: object stored in the file
    :rtype: Any
    """
    data = joblib.load(path)
    logger.info(f"binary file: {path} loaded successfully")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    :param path: path to the file
    :type path: Path
    :return: size in KB
    :rtype: str
    """
    size = round(os.path.getsize(path) / 1024)
    return f"~ {size} KB"

@ensure_annotations
def decode_image(img_string: str, file_name: Path):
    """decode image

    :param img_string: image string
    :type img_string: str
    :param file_name: path to the image file
    :type file_name: Path
    """
    img_data = base64.b64decode(img_string)
    with open(file_name, "wb") as f:
        f.write(img_data)
        logger.info(f"image file: {file_name} saved successfully")

@ensure_annotations
def encode_image_to_base64(file_name: Path) -> str:
    """encode image to base64

    :param file_name: path to the image file
    :type file_name: Path
    :return: encoded image string
    :rtype: str
    """
    with open(file_name, "rb") as f:
        img_string = base64.b64encode(f.read())
        logger.info(f"image file: {file_name} loaded successfully")
        return img_string


@ensure_annotations
def check_file_exists(file_name: str) -> bool:
    """Check if file exists

    :param file_name: path to the file
    :type file_name: Path
    :return: file exists or not
    :rtype: bool
    """
    return os.path.exists(file_name)
