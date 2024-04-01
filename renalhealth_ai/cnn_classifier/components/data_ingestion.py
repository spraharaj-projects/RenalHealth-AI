import os
import zipfile
import gdown
from cnn_classifier.utils.common import get_size, check_file_exists
from cnn_classifier.entity.config_entity import DataIngestionConfig
from cnn_classifier import logger


class DataIngestion:
  def __init__(self, config: DataIngestionConfig):
    self.config = config

  def download_data(self) -> str:
    """fetch data from the url

    :return: _description_
    :rtype: str
    """
    if check_file_exists(self.config.local_data_dir):
      logger.info(f"File already available at: {self.config.local_data_dir}")
    else:  
      try:
        dataset_url = self.config.source_URL
        zip_download_dir = self.config.local_data_dir
        root_dir = self.config.root_dir
        os.makedirs(root_dir, exist_ok=True)
        logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

        file_id = dataset_url.split("/")[-2]
        prefix = "https://drive.google.com/uc?/export=download&id="
        gdown.download(prefix + file_id, zip_download_dir)

        logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
      
      except Exception as e:
        raise e
  
  def unzip_data(self):
    """extract the zip file into the data dictionary
    """
    unzip_path = self.config.unzip_dir
    os.makedirs(unzip_path,exist_ok=True)
    with zipfile.ZipFile(self.config.local_data_dir, "r") as zip_ref:
      zip_ref.extractall(unzip_path)
