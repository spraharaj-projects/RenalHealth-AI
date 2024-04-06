from torchvision import models
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch import (
    device as torch_device,
    cuda as torch_cuda,
    max as torch_max,
    load as torch_load,
    float32 as torch_float32,
    no_grad,
    nn,
)
from torch.optim import SGD
from torch.utils.data import DataLoader, SubsetRandomSampler
import mlflow
from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm

from cnn_classifier import logger
from cnn_classifier.entity.config_entity import EvaluationConfig
from cnn_classifier.utils.common import save_json


class Evaluation:
    """
    Class responsible for model evaluation.

    :param config: The configuration for evaluation.
    :type config: :class:`EvaluationConfig`

    **Attributes:**

    - ``config``: The configuration for evaluation.
    - ``device``: The device (CPU or GPU) on which the evaluation will be performed.
    - ``model``: The model to be evaluated.
    - ``valid_loader``: DataLoader for validation dataset.
    - ``criterion``: The loss function.
    - ``optimizer``: The optimizer.
    - ``score``: Evaluation score (loss and accuracy).

    **Methods:**

    - :meth:`load_model`: Loads the model and its state dictionary.
    - :meth:`_valid_generator`: Prepares the data loader for validation.
    - :meth:`validate`: Validates the model.
    - :meth:`compile`: Compiles the model for evaluation.
    - :meth:`evaluation`: Performs the evaluation process.
    - :meth:`save_score`: Saves the evaluation score to a JSON file.
    - :meth:`log_into_mlflow`: Logs evaluation metrics into MLFlow.

    **Example Usage:**

    .. code-block:: python

        # Create Evaluation instance
        evaluation = Evaluation(evaluation_config)

        # Perform model evaluation
        evaluation.evaluation()

        # Save evaluation score
        evaluation.save_score()

        # Log evaluation metrics into MLFlow
        evaluation.log_into_mlflow()
    """
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self.device = torch_device(
            'cuda' if torch_cuda.is_available() else 'cpu'
        )
        logger.info(f"Evaluation initialized on device: {self.device}")

    def _valid_generator(self) -> None:
        """
        Prepares the data loader for validation.
        """
        valid_transform = v2.Compose([
            v2.Resize(self.config.params_image_size[:-1]),
            v2.ToImage(),
            v2.ToDtype(torch_float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = ImageFolder(root=self.config.training_data)
        
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(0.3 * num_train)
        _, valid_indices = indices[split:], indices[:split]

        valid_sampler = SubsetRandomSampler(valid_indices)

        self.valid_loader = DataLoader(
            dataset,
            batch_size=self.config.params_batch_size,
            sampler=valid_sampler,
            num_workers=8,
            pin_memory=True,
        )

        self.valid_loader.dataset.transform = valid_transform
    
    @staticmethod
    def load_model(model_path: Path, model_state_dict_path: Path) -> models:
        """Loads the model and its state dictionary.

        :param model_path: The path to the model file.
        :type model_path: :class:`Path`
        :param model_state_dict_path: The path to the model's state dictionary
        file.
        :type model_state_dict_path: :class:`Path`
        :return: The loaded model.
        :rtype: :class:`torchvision.models`
        """
        model = torch_load(model_path)
        model_state_dict = torch_load(model_state_dict_path)
        model.load_state_dict(model_state_dict)
        return model

    def validate(self) -> None:
        """Validates the model.

        :return: Evaluation score (loss and accuracy).
        :rtype: list[float, float]
        """      
        self.model.eval()
        
        valid_running_loss = 0.0
        valid_correct = 0
        valid_total = 0

        logger.info("Validation start")

        with no_grad():
            with tqdm(self.valid_loader, unit="batch") as tepoch:
                tepoch.set_description("Validation")
                for data, target in tepoch:
                    inputs, labels = data.to(self.device), target.to(self.device)
                    outputs = self.model(inputs)
                    
                    loss = self.criterion(outputs, labels)
                    
                    valid_running_loss += loss.item()

                    _, predicted = torch_max(outputs.data, 1)
                    valid_total += labels.size(0)
                    valid_correct += (predicted == labels).sum().item()
                    tepoch.set_postfix_str(
                        f"Loss: {loss.item() / 100:.3f}, "
                        f"Running Loss: {valid_running_loss / 100:.3f}, "
                        f"Accuracy: {(valid_correct / valid_total) * 100:.3f}"
                    )

        return [
            valid_running_loss / len(self.valid_loader.dataset),
            (valid_correct / valid_total) * 100,
        ]
    
    def compile(self) -> None:
        """
        Compiles the model for evaluation.
        """
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = SGD(
            self.model.parameters(),
            lr=self.config.params_learning_rate
        )

    def evaluation(self) -> None:
        """
        Performs the evaluation process.
        """
        self.model = self.load_model(
            self.config.model_path,
            self.config.model_state_dict_path).to(self.device)
        self._valid_generator()
        self.compile()
        self.score = self.validate()
        self.save_score()
    
    def save_score(self) -> None:
        """
        Saves the evaluation score to a JSON file.
        """
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
    
    def log_into_mlflow(self) -> None:
        """
        Logs evaluation metrics into MLFlow.
        """
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {
                    "loss": self.score[0],
                    "accuracy": self.score[1],
                }
            )
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Register, which depends
                # on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                logger.info(f"Logging model into MLFlow UI")
                mlflow.pytorch.log_model(
                    self.model,
                    "model",
                    registered_model_name="VGG16Model",
                )
            else:
                logger.info(f"MLFlow Local Logging")
                mlflow.pytorch.log_model(self.model, "model")
