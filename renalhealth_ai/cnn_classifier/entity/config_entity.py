from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
  """
  Class to hold the data ingestion configuration parameters
  """
  root_dir: Path
  source_URL: str
  local_data_dir: Path
  unzip_dir: Path
