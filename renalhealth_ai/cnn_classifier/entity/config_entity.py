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
    """
    Class to hold the prepare base model configuration parameters
    """
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: List[int]
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int


@dataclass(frozen=True)
class TrainingConfig:
    """
    Class to hold the training configuration parameters
    """
    root_dir: Path
    trained_model_state_dict_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_agumentation: bool
    params_image_size: List[int]
    params_learning_rate: float


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Class to hold the evaluation configuration parameters
    """
    model_path: Path
    model_state_dict_path: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
    params_learning_rate: int
