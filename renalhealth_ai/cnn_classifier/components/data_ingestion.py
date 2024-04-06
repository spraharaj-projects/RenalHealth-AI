import os
import zipfile
import gdown
from cnn_classifier.utils.common import get_size, check_file_exists
from cnn_classifier.entity.config_entity import DataIngestionConfig
from cnn_classifier import logger


class DataIngestion:
    """Class responsible for downloading and extracting data.

      :param config: The configuration for data ingestion.
      :type config: :class:`DataIngestionConfig`

      **Attributes:**

      - ``config``: The data ingestion configuration.

      **Methods:**

      - :meth:`download_data`: Fetches data from the URL.
      - :meth:`unzip_data`: Extracts the zip file into the data directory.

      **Example Usage:**

      .. code-block:: python

          # Create DataIngestion instance
          data_ingestion = DataIngestion(data_ingestion_config)

          # Download data
          data_ingestion.download_data()

          # Unzip data
          data_ingestion.unzip_data()
      """

    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def download_data(self) -> str:
        """Fetches data from the URL.

        :return: A message indicating the status of the download.
        :rtype: str
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_dir
            root_dir = self.config.root_dir
            os.makedirs(root_dir, exist_ok=True)
            logger.info(
                f"Downloading data from {dataset_url} " 
                f"into file {zip_download_dir}"
            )

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix + file_id, zip_download_dir)

            logger.info(
                f"Downloaded data from {dataset_url} "
                f"into file {zip_download_dir}"
            )

        except Exception as e:
            raise e

    def unzip_data(self) -> None:
        """
        Extracts the zip file into the data directory.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_dir, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
