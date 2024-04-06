from pathlib import Path
from torch import (
    device as torch_device,
    cuda as torch_cuda,
    load as torch_load,
    save as torch_save,
    max as torch_max,
    float32 as torch_float32,
    nn,
    no_grad,
)
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import SGD
from torchvision import models
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from cnn_classifier import logger
from cnn_classifier.entity.config_entity import TrainingConfig


class Training:
    """Class responsible for training the model.

    :param config: The configuration for training.
    :type config: :class:`TrainingConfig`

    **Attributes:**

    - ``config``: The configuration for training.
    - ``device``: The device (CPU or GPU) on which the model will be trained.
    - ``model``: The model to be trained.
    - ``criterion``: The loss function.
    - ``optimizer``: The optimizer.

    **Methods:**

    - :meth:`get_base_model`: Loads the pre-trained base model.
    - :meth:`train_valid_generator`: Prepares the data loaders for training and validation.
    - :meth:`save_model`: Saves the trained model.
    - :meth:`compile`: Compiles the model.
    - :meth:`train`: Trains the model for one epoch.
    - :meth:`validate`: Validates the trained model.
    - :meth:`process`: Starts the training process.

    **Example Usage:**

    .. code-block:: python

        # Create Training instance
        training = Training(training_config)

        # Load the pre-trained base model
        training.get_base_model()

        # Prepare data loaders for training and validation
        training.train_valid_generator()

        # Compile the model
        training.compile()

        # Start the training process
        training.process()
    """
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch_device(
            "cuda" if torch_cuda.is_available() else "cpu"
        )
        logger.info(f"Training initialized on device: {self.device}")
    
    def get_base_model(self) -> None:
        """
        Loads the pre-trained base model.
        """
        self.model = torch_load(
            self.config.updated_base_model_path
        ).to(self.device)

    def train_valid_generator(self) -> None:
        """
        Prepares the data loaders for training and validation.
        """
        train_transform = v2.Compose([
            v2.Resize(self.config.params_image_size[:-1]),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(40),
            v2.RandomAffine(
                degrees=0,
                translate=(0.2, 0),
                shear=0.2,
                scale=(0.8, 1.2),
            ),
            v2.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.2
            ),
            v2.ToImage(),
            v2.ToDtype(torch_float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        valid_transform = v2.Compose([
            v2.Resize(self.config.params_image_size[:-1]),
            v2.ToImage(),
            v2.ToDtype(torch_float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if self.config.params_is_agumentation:
            train_transform = valid_transform
        
        dataset = ImageFolder(root=self.config.training_data)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(0.2 * num_train)
        train_indices, valid_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.params_batch_size,
            sampler=train_sampler,
            num_workers=8,
            pin_memory=True,
        )
        self.valid_loader = DataLoader(
            dataset,
            batch_size=self.config.params_batch_size,
            sampler=valid_sampler,
            num_workers=8,
            pin_memory=True,
        )
        
        self.train_loader.dataset.transform = train_transform
        self.valid_loader.dataset.transform = valid_transform
    
    @staticmethod
    def save_model_state_dict(path: Path, model: models) -> None:
        """
        Saves the trained model.

        :param path: The path where the model will be saved.
        :type path: :class:`Path`
        :param model: The trained model to be saved.
        :type model: :class:`torchvision.models`
        """
        torch_save(model.state_dict(), path)
        logger.info(f"Model saved to: {path}")
    
    def compile(self) -> None:
        """
        Compiles the model.
        """
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = SGD(
            self.model.parameters(),
            lr=self.config.params_learning_rate
        )
        logger.info("Model criterion and optimizer prepared")
    
    def train(self, epoch: int) -> None:
        """Trains the model for one epoch.

        :param epoch: The current epoch number.
        :type epoch: int
        """
        self.model.train()
        
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        logger.info(f"Training start for epoch {epoch + 1}")

        with tqdm(self.train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")

                inputs, labels = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_running_loss += loss.item()

                _, predicted = torch_max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                tepoch.set_postfix_str(
                    f"Loss: {loss.item() / 100:.3f}, "
                    f"Running Loss: {train_running_loss / 100:.3f}, "
                    f"Accuracy: {(train_correct / train_total) * 100:.3f}"
                )
    
    def validate(self, epoch: int) -> None:
        """Validates the trained model.

        :param epoch: The current epoch number.
        :type epoch: int
        """
        self.model.eval()
        
        valid_running_loss = 0.0
        valid_correct = 0
        valid_total = 0

        logger.info(f"Validation start for epoch: {epoch + 1}")

        with no_grad():
            with tqdm(self.valid_loader, unit="batch") as tepoch:
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

        logger.info(
            f"[Epoch {epoch + 1}]: "
            f"Valid Loss: {valid_running_loss / len(self.valid_loader.dataset):.3f}, "
            f"Valid Accuracy: {(valid_correct / valid_total) * 100:.3f}"
        )

    def process(self) -> None:
        """
        Starts the training process.
        """
        for epoch in range(self.config.params_epochs):
            self.train(epoch)
            self.validate(epoch)
        logger.info("Training Completed")
        self.save_model_state_dict(
            path=self.config.trained_model_state_dict_path,
            model=self.model,
        )
