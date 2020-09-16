from typing import Callable, Dict, Tuple
import torch
import numpy as np

from detector.models.base import Model


class PredictorModel(Model):
    """CharacterModel works on datasets providing images, with one-hot labels."""

    def __init__(self, network_fn, network_args : Dict = None):
        super().__init__(network_fn, network_args)

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        x = self.preprocess(image)
        return self.predict_on_tensor(x)

    def predict_on_tensor(self, image: torch.tensor) -> Tuple[str, float]:
        x = image
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 0)
        pred_raw = self.network(x).flatten().detach().numpy()
        ind = np.argmax(pred_raw)
        return ind