import os
from pathlib import Path

from cnn_classifier.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
)
from cnn_classifier.utils.common import read_yaml, create_directories
from cnn_classifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig,
)


class ConfigurationManager:
    """
    Class responsible for managing and retrieving configurations for data 
    ingestion, model preparation, training, and evaluation.

    :param config_file_path: The path to the YAML configuration file containing 
    general configurations.
    :type config_file_path: :class:`Path`
    :param params_file_path: The path to the YAML file containing parameters 
    specific to the models.
    :type params_file_path: :class:`Path`

    **Attributes:**

    - ``config``: A dictionary containing general configurations loaded from the
    YAML file.
    - ``params``: A dictionary containing model-specific parameters loaded from
    the YAML file.

    **Methods:**

    - :meth:`get_data_ingestion_config`: Retrieves the configuration for data 
    ingestion.
    - :meth:`get_prepare_base_model_config`: Retrieves the configuration for 
    preparing the base model.
    - :meth:`get_training_config`: Retrieves the configuration for training the 
    model.
    - :meth:`get_evaluation_config`: Retrieves the configuration for evaluating 
    the model.

    **Example Usage:**

    .. code-block:: python

        # Create ConfigurationManager instance
        config_manager = ConfigurationManager()

        # Get data ingestion configuration
        data_ingestion_config = config_manager.get_data_ingestion_config()

        # Get prepare base model configuration
        prepare_base_model_config = config_manager.get_prepare_base_model_config()

        # Get training configuration
        training_config = config_manager.get_training_config()

        # Get evaluation configuration
        evaluation_config = config_manager.get_evaluation_config()
    """

    def __init__(
        self,
        config_file_path: Path = CONFIG_FILE_PATH,
        params_file_path: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves the configuration for data ingestion.

        :return: The data ingestion configuration.
        :rtype: :class:`DataIngestionConfig`
        """
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_dir=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """
        Retrieves the configuration for preparing the base model.

        :return: The prepare base model configuration.
        :rtype: :class:`PrepareBaseModelConfig`
        """
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
        )

        return prepare_base_model_config

    def get_training_config(self) -> TrainingConfig:
        """
        Retrieves the configuration for training the model.

        :return: The training configuration.
        :rtype: :class:`TrainingConfig`
        """
        training = self.config.training
        updated_base_model_path = self.config.prepare_base_model.updated_base_model_path
        unzip_dir = self.config.data_ingestion.unzip_dir
        params = self.params
        training_data = os.path.join(
            unzip_dir,
            "kidney-dataset",
        )
        create_directories([training.root_dir])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_state_dict_path=Path(
                training.trained_model_state_dict_path),
            updated_base_model_path=Path(updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_agumentation=params.AGUMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        """
        Retrieves the configuration for model evaluation.

        This method retrieves the configuration required for evaluating the 
        model, including the paths to the updated base model, trained model 
        state dictionary, and training data, as well as additional parameters 
        for evaluation.

        :return: The evaluation configuration.
        :rtype: :class:`EvaluationConfig`
        """
        updated_base_model_path = Path(
            self.config.prepare_base_model.updated_base_model_path
        )
        trained_model_state_dict_path = Path(
            self.config.training.trained_model_state_dict_path
        )
        training_data = os.path.join(
            self.config.data_ingestion.unzip_dir,
            "kidney-dataset",
        )

        eval_config = EvaluationConfig(
            model_path=updated_base_model_path,
            model_state_dict_path=trained_model_state_dict_path,
            training_data=training_data,
            mlflow_uri="https://dagshub.com/spraharaj-projects/RenalHealth-AI.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
        )

        return eval_config
