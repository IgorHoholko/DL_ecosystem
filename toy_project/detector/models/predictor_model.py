from typing import Callable, Dict, Tuple

import numpy as np

from detector.models.base import Model


class PredictorModel(Model):
    """CharacterModel works on datasets providing images, with one-hot labels."""

    def __init__(self, network_fn, network_args : Dict = None):
        super().__init__(network_fn, network_args)

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        x = self.preprocess(image)
        pred_raw = self.network(x).flatten().detach().numpy()
        ind = np.argmax(pred_raw)
        return ind