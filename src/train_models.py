"""
Train baseline and tree-based machine learning models (XGBoost and LightGBM) for the eight targets.

The script reuses the same time-aware train/test split for each model family.
It trains the primary fantasy-points target first, then loops through the
remaining targets. Metrics are compared against a simple train-mean baseline so
each model has an easy reference point.

For each target, the baseline predicts the training-set average for every test
row, then calculates MAE, RMSE, and R2 against the actual test values.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_modelling import MODELLING_DIR, SPLIT_DIR, TARGET_COLUMNS

PRIMARY_TARGET = "FANTASY_PTS"
DEFAULT_MODEL_NAME = "xgboost"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    prediction_column: str
    metric_prefix: str
    model_extension: str

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / "models" / self.name

    @property
    def output_dir(self) -> Path:
        return MODELLING_DIR / self.name / "test_predictions"


MODEL_CONFIGS = {
    "xgboost": ModelConfig(
        name="xgboost",
        prediction_column="xgboost_prediction",
        metric_prefix="xgboost",
        model_extension="json",
    ),
    "lightgbm": ModelConfig(
        name="lightgbm",
        prediction_column="lightgbm_prediction",
        metric_prefix="lightgbm",
        model_extension="txt",
    ),
}

MODELS_DIR = MODEL_CONFIGS[DEFAULT_MODEL_NAME].models_dir
TEST_PREDICTIONS_DIR = MODEL_CONFIGS[DEFAULT_MODEL_NAME].output_dir


def get_model_config(model_name: str) -> ModelConfig:
    try:
        return MODEL_CONFIGS[model_name]
    except KeyError as error:
        valid_models = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {valid_models}.") from error


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


def create_model(model_name: str):
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(random_state=42, n_jobs=-1)

    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)

    raise ValueError(f"Unsupported model: {model_name}")


def train_model(model_name: str, x_train: pd.DataFrame, y_train: pd.Series):
    model = create_model(model_name)
    model.fit(x_train, y_train)
    return model


def model_path_for_target(target: str, config: ModelConfig, models_dir: Path | None = None) -> Path:
    base_dir = config.models_dir if models_dir is None else models_dir
    return base_dir / f"{config.name}_{target.lower()}.{config.model_extension}"


def save_model(model, model_path: Path, config: ModelConfig) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if config.name == "lightgbm":
        model.booster_.save_model(str(model_path))
    else:
        model.save_model(model_path)


def train_target_model(
    target: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelConfig,
    models_dir: Path | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    baseline_predictions = baseline_mean_predictions(y_train[target], len(y_test))
    baseline_metrics = calculate_regression_metrics(y_test[target], baseline_predictions)

    model = train_model(config.name, x_train, y_train[target])
    model_predictions = model.predict(x_test)
    model_metrics = calculate_regression_metrics(y_test[target], model_predictions)

    model_path = model_path_for_target(target, config, models_dir=models_dir)
    save_model(model, model_path, config)

    metrics = {
        "target": target,
        "baseline_mae": baseline_metrics["mae"],
        "baseline_rmse": baseline_metrics["rmse"],
        "baseline_r2": baseline_metrics["r2"],
        f"{config.metric_prefix}_mae": model_metrics["mae"],
        f"{config.metric_prefix}_rmse": model_metrics["rmse"],
        f"{config.metric_prefix}_r2": model_metrics["r2"],
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
    }
    predictions = pd.DataFrame(
        {
            "target": target,
            "actual": y_test[target].to_numpy(),
            "baseline_prediction": baseline_predictions,
            config.prediction_column: model_predictions,
        }
    )
    return metrics, predictions


def train_all_targets(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    targets: list[str] = TARGET_COLUMNS,
    config: ModelConfig | None = None,
    models_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_config = MODEL_CONFIGS[DEFAULT_MODEL_NAME] if config is None else config
    ordered_targets = [PRIMARY_TARGET] + [target for target in targets if target != PRIMARY_TARGET]
    all_metrics = []
    all_predictions = []

    for index, target in enumerate(ordered_targets, start=1):
        print(f"Training {model_config.name} {index}/{len(ordered_targets)}: {target}")
        metrics, predictions = train_target_model(
            target,
            x_train,
            x_test,
            y_train,
            y_test,
            config=model_config,
            models_dir=models_dir,
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    return pd.DataFrame(all_metrics), pd.concat(all_predictions, ignore_index=True)


def write_training_outputs(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path = TEST_PREDICTIONS_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "model_predictions.parquet", index=False)
    with (output_dir / "model_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics.to_dict(orient="records"), file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline and tree-based models for NBA targets.")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default=DEFAULT_MODEL_NAME)
    parser.add_argument("--input-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_model_config(args.model)
    output_dir = config.output_dir if args.output_dir is None else args.output_dir
    models_dir = config.models_dir if args.models_dir is None else args.models_dir

    x_train, x_test, y_train, y_test = load_modelling_matrices(args.input_dir)
    metrics, predictions = train_all_targets(
        x_train,
        x_test,
        y_train,
        y_test,
        config=config,
        models_dir=models_dir,
    )

    print("\nModel Metrics")
    print("-------------")
    print(metrics.drop(columns=["model_path"]).round(4).to_string(index=False))

    if not args.no_export:
        write_training_outputs(metrics, predictions, output_dir)
        print(f"\nSaved metrics and predictions to {output_dir}")
        print(f"Saved models to {models_dir}")


if __name__ == "__main__":
    main()
