from pathlib import Path
from typing import Dict, Callable
import numpy as np
import importlib
from torchvision import transforms

import torch
from torch.utils.data import DataLoader
from detector.datasets import DatasetSequence

DIRNAME = Path(__file__).parents[1].resolve() / "weights"


class Model(torch.nn.Module):
    """Base class, to be subclassed by predictors for specific type of data."""

    def __init__(self,
        network_fn,
        network_args: Dict = None,
    ):
        super(Model, self).__init__()

        models_module = importlib.import_module("detector.networks")
        model_class_ = getattr(models_module, network_fn)
        self.name = f"{self.__class__.__name__}_{model_class_.__name__}"
        if network_args is None:
            network_args = {}

        self.network = model_class_(**network_args)
        print(self.network)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])

    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f"{self.name}_weights.h5")

    def preprocess(self, image: np.array):
        x = self.transforms(image)
        x = torch.unsqueeze(x, 0)
        return x

    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = 16):
        sequence = DatasetSequence(x, y)
        loader = DataLoader(dataset=sequence,
                             batch_size=batch_size,
                             shuffle=False, )

        preds = self.network(x)
        return np.mean(np.argmax(preds, -1) == np.argmax(y, -1))

    def loss(self):
        return "CrossEntropyLoss"

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)