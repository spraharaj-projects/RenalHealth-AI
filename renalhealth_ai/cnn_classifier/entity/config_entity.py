from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class DataIngestionConfig:
  """
  Class to hold the data ingestion configuration parameters
  """
  root_dir: Path
  source_URL: str
  local_data_dir: Path
  unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
  root_dir: Path
  base_model_path: Path
  updated_base_model_path: Path
  params_image_size: List[int]
  params_learning_rate: float
  params_include_top: bool
  params_weights: str
  params_classes: int
