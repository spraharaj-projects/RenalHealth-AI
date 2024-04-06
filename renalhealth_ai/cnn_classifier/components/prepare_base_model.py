from pathlib import Path
from typing import List
from torchvision import models
from torch import (
    device as torch_device,
    cuda as torch_cuda,
    save as torch_save,
    nn,
)
from torchsummary import summary
from cnn_classifier.entity.config_entity import PrepareBaseModelConfig
from cnn_classifier import logger


class PrepareBaseModel:
    """
    Class responsible for preparing and updating the base model.

    :param config: The configuration for preparing the base model.
    :type config: :class:`PrepareBaseModelConfig`

    **Attributes:**

    - ``config``: The configuration for preparing the base model.
    - ``device``: The device (CPU or GPU) on which the model will be trained.

    **Methods:**

    - :meth:`get_base_model`: Retrieves the base model specified in the configuration.
    - :meth:`update_base_model`: Updates the base model with additional layers and saves the updated model.
    - :meth:`_prepare_full_model`: Prepares the full model by adding custom layers.
    - :meth:`save_model`: Saves the model to the specified path.

    **Example Usage:**

    .. code-block:: python

        # Create PrepareBaseModel instance
        prepare_base_model = PrepareBaseModel(prepare_base_model_config)

        # Get the base model
        prepare_base_model.get_base_model()

        # Update the base model
        prepare_base_model.update_base_model()
    """
    def __init__(self, config: PrepareBaseModelConfig) -> None:
        self.config = config
        self.device = torch_device(
            "cuda" if torch_cuda.is_available() else "cpu"
        )
        logger.info(f"Prepare Base Model initialized on device: {self.device}")

    def get_base_model(self) -> None:
        """
        Retrieves the base model specified in the configuration.
        """
        if self.config.params_weights == "imagenet":
            self.model = models.vgg16(
                weights=models.VGG16_Weights.IMAGENET1K_V1
            ).to(self.device)
        if not self.config.params_include_top:
            self.model.classifier = nn.Sequential(
                *list(self.model.classifier.children())[:-1]
            )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(
        model: models,
        classes: int,
        freeze_all: bool,
        freeze_till: int,
        img_size: List[int],
        device: str,
    ) -> models:
        """
        Prepares the full model by adding custom layers.

        :param model: The base model.
        :type model: :class:`torchvision.models`
        :param classes: The number of output classes.
        :type classes: int
        :param freeze_all: Whether to freeze all layers.
        :type freeze_all: bool
        :param freeze_till: The index till which layers should be frozen.
        :type freeze_till: int or None
        :param img_size: The dimensions of input images for the model.
        :type img_size: list[int]
        :param device: The device (CPU or GPU) on which the model will be 
        trained.
        :type device: :class:`torch.device`
        :return: The prepared full model.
        :rtype: :class:`torchvision.models`
        """
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for param in model.parameters()[:freeze_till]:
                param.requires_grad = False

        last_layer = None
        for layer in model.classifier.children():
            if isinstance(layer, nn.Linear):
                last_layer = layer
        if last_layer is None:
            raise ValueError("No linear layer found in the classifier.")
        num_features = last_layer.in_features

        model.classifier.append(nn.Linear(num_features, classes).to(device))
        model.classifier.append(nn.Softmax(dim=1).to(device))

        print(model)
        summary(model, tuple(reversed(img_size)))
        return model

    def update_base_model(self) -> None:
        """
        Updates the base model with additional layers and saves the updated
        model.
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            img_size=self.config.params_image_size,
            device=self.device,
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: models) -> None:
        """
        Saves the model to the specified path.

        :param path: The path where the model will be saved.
        :type path: :class:`Path`
        :param model: The model to be saved.
        :type model: :class:`torchvision.models`
        """
        torch_save(model, path)
