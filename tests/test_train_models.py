from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.train_models import baseline_mean_predictions, calculate_regression_metrics


class TrainModelHelperTests(unittest.TestCase):
    def test_calculate_regression_metrics_for_perfect_predictions(self) -> None:
        y_true = pd.Series([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])

        metrics = calculate_regression_metrics(y_true, y_pred)

        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["r2"], 1.0)

    def test_baseline_mean_predictions_use_training_target_mean(self) -> None:
        y_train = pd.Series([10.0, 20.0, 30.0])

        predictions = baseline_mean_predictions(y_train, row_count=4)

        self.assertEqual(predictions.tolist(), [20.0, 20.0, 20.0, 20.0])


if __name__ == "__main__":
    unittest.main()
