from torchvision import datasets
from pathlib import Path
import torchvision.models as models
import numpy as np

class Model():
    def __init__(self, model_name, num_classes):
        self.name = model_name
        self.num_classes = num_classes

    def get_model(self):
        model = getattr(models, self.name)(num_classes=self.num_classes)

        return model