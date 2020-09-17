"""Run validation test for PredictorModel_LeNet."""

from time import time
import unittest
import torch

from detector.datasets import SVHNDataset
from detector.models import PredictorModel



class TestEvaluatePredictorModel_LeNet(unittest.TestCase):
    def test_evaluate(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        predictor = PredictorModel('LeNet')
        predictor.load_weights()
        predictor.eval()
        predictor.to(device)

        dataset = SVHNDataset()
        dataset.load_or_generate_data()

        t = time()
        metric = predictor.evaluate(dataset.x_test, dataset.y_test,
                                    device=device)

        time_taken = time() - t
        print(f"acc: {metric}, time_taken: {time_taken} sec")
        self.assertGreater(metric, 0.6)
        self.assertLess(time_taken, 10)