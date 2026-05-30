"""
Train baseline and XGBoost models for the eight prediction targets.

The script starts with the primary fantasy-points target, then trains the same
default XGBoost setup for the remaining targets. Metrics are compared against a
simple train-mean baseline so model performance has an easy reference point.

For each target, the baseline predicts the training-set average for every test
row, then calculates MAE, RMSE, and R2 against the actual test values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_modelling import MODELLING_DIR, SPLIT_DIR, TARGET_COLUMNS

MODELS_DIR = PROJECT_ROOT / "models"
PRIMARY_TARGET = "FANTASY_PTS"


def load_modelling_matrices(input_dir: Path = SPLIT_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_train = pd.read_parquet(input_dir / "X_train.parquet")
    x_test = pd.read_parquet(input_dir / "X_test.parquet")
    y_train = pd.read_parquet(input_dir / "y_train.parquet")
    y_test = pd.read_parquet(input_dir / "y_test.parquet")
    return x_train, x_test, y_train, y_test


def calculate_regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    total_sum_squares = float(np.sum((actual - np.mean(actual)) ** 2))
    residual_sum_squares = float(np.sum(errors**2))
    r2 = 0.0 if total_sum_squares == 0 else float(1 - residual_sum_squares / total_sum_squares)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def baseline_mean_predictions(y_train: pd.Series, row_count: int) -> np.ndarray:
    return np.full(row_count, y_train.mean(), dtype=float)


def train_xgboost_model(x_train: pd.DataFrame, y_train: pd.Series):
    from xgboost import XGBRegressor

    model = XGBRegressor(random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    return model


def train_target_model(
    target: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    models_dir: Path = MODELS_DIR,
) -> tuple[dict[str, object], pd.DataFrame]:
    baseline_predictions = baseline_mean_predictions(y_train[target], len(y_test))
    baseline_metrics = calculate_regression_metrics(y_test[target], baseline_predictions)

    model = train_xgboost_model(x_train, y_train[target])
    xgboost_predictions = model.predict(x_test)
    xgboost_metrics = calculate_regression_metrics(y_test[target], xgboost_predictions)

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"xgboost_{target.lower()}.json"
    model.save_model(model_path)

    metrics = {
        "target": target,
        "baseline_mae": baseline_metrics["mae"],
        "baseline_rmse": baseline_metrics["rmse"],
        "baseline_r2": baseline_metrics["r2"],
        "xgboost_mae": xgboost_metrics["mae"],
        "xgboost_rmse": xgboost_metrics["rmse"],
        "xgboost_r2": xgboost_metrics["r2"],
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
    }
    predictions = pd.DataFrame(
        {
            "target": target,
            "actual": y_test[target].to_numpy(),
            "baseline_prediction": baseline_predictions,
            "xgboost_prediction": xgboost_predictions,
        }
    )
    return metrics, predictions


def train_all_targets(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    targets: list[str] = TARGET_COLUMNS,
    models_dir: Path = MODELS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_targets = [PRIMARY_TARGET] + [target for target in targets if target != PRIMARY_TARGET]
    all_metrics = []
    all_predictions = []

    for index, target in enumerate(ordered_targets, start=1):
        print(f"Training {index}/{len(ordered_targets)}: {target}")
        metrics, predictions = train_target_model(target, x_train, x_test, y_train, y_test, models_dir=models_dir)
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    return pd.DataFrame(all_metrics), pd.concat(all_predictions, ignore_index=True)


def write_training_outputs(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path = MODELLING_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "model_predictions.parquet", index=False)
    with (output_dir / "model_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics.to_dict(orient="records"), file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline and XGBoost models for NBA targets.")
    parser.add_argument("--input-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=MODELLING_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x_train, x_test, y_train, y_test = load_modelling_matrices(args.input_dir)
    metrics, predictions = train_all_targets(x_train, x_test, y_train, y_test, models_dir=args.models_dir)

    print("\nModel Metrics")
    print("-------------")
    print(metrics.drop(columns=["model_path"]).round(4).to_string(index=False))

    if not args.no_export:
        write_training_outputs(metrics, predictions, args.output_dir)
        print(f"\nSaved metrics and predictions to {args.output_dir}")
        print(f"Saved models to {args.models_dir}")


if __name__ == "__main__":
    main()
