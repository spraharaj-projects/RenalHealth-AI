from pathlib import Path
from torchvision import models
from torch import (
    device as torch_device,
    cuda as torch_cuda,
    save as torch_save,
    nn,
)
from torchsummary import summary
from cnn_classifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.device = torch_device(
            "cuda" if torch_cuda.is_available() else "cpu"
        )

    def get_base_model(self):
        self.model = models.vgg16(
            weights=self.config.params_weights
        ).to(self.device)
        if not self.config.params_include_top:
            self.model.classifier = nn.Sequential(
                *list(self.model.classifier.children())[:-1]
            )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(
        model,
        classes,
        freeze_all,
        freeze_till,
        img_size,
        device,
    ):
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

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            img_size=self.config.params_image_size,
            device=self.device,
        )

        self.save_model(
            path=self.config.updated_base_model_path,
            model=self.full_model
        )

    @staticmethod
    def save_model(path: Path, model: models):
        torch_save(model, path)
