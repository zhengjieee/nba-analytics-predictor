"""
Prepare time-aware train/test data for modelling.

It loads the completed feature table, sorts games chronologically, uses
the most recent portion of games as the test set, and separates model features
from the eight prediction targets. 

Same-game box-score outcomes are excluded from X to avoid training on information that would only be known after a game, preventing data leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_FULL_PATH = PROCESSED_DIR / "features_full.parquet"
MODELLING_DIR = PROCESSED_DIR / "modelling"

TARGET_COLUMNS = ["FANTASY_PTS", "PTS", "REB", "AST", "STL", "BLK", "TOV", "3PM"]

IDENTIFIER_COLUMNS = [
    "SEASON_ID",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "PLAYER_SLUG",
    "opponent_team_id",
]

CATEGORICAL_SOURCE_COLUMNS = [
    "WL",
    "SEASON",
    "BIRTHDATE",
    "SCHOOL",
    "COUNTRY",
    "HEIGHT",
    "POSITION",
    "ROSTERSTATUS",
    "DRAFT_YEAR",
    "DRAFT_ROUND",
    "DRAFT_PICK",
    "opponent_abbreviation",
    "days_rest_category",
]

SAME_GAME_RESULT_COLUMNS = [
    "MIN",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "PF",
    "PLUS_MINUS",
    "VIDEO_AVAILABLE",
    "true_shooting_pct",
    "effective_fg_pct",
    "three_point_attempt_rate",
    "free_throw_attempt_rate",
    "fantasy_pts_change_from_previous",
    "pts_change_from_previous",
    "reb_change_from_previous",
    "ast_change_from_previous",
    "stl_change_from_previous",
    "blk_change_from_previous",
    "tov_change_from_previous",
    "3pm_change_from_previous",
]


def load_completed_features(path: Path = FEATURES_FULL_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def sort_by_game_date(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.copy()
    sorted_df["GAME_DATE"] = pd.to_datetime(sorted_df["GAME_DATE"])
    return sorted_df.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True)


def time_series_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    sorted_df = sort_by_game_date(df)
    split_index = int(len(sorted_df) * (1 - test_size))
    split_index = min(max(split_index, 1), len(sorted_df) - 1)
    cutoff_date = sorted_df.loc[split_index, "GAME_DATE"]

    train = sorted_df[sorted_df["GAME_DATE"] < cutoff_date].copy()
    test = sorted_df[sorted_df["GAME_DATE"] >= cutoff_date].copy()

    if train.empty or test.empty:
        raise ValueError("Time split produced an empty train or test set.")
    if train["GAME_DATE"].max() >= test["GAME_DATE"].min():
        raise ValueError("Train/test split has overlapping dates.")

    return train, test


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded_columns = set(
        TARGET_COLUMNS
        + IDENTIFIER_COLUMNS
        + CATEGORICAL_SOURCE_COLUMNS
        + SAME_GAME_RESULT_COLUMNS
    )
    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns
    return [
        column
        for column in numeric_columns
        if column not in excluded_columns
    ]


def create_feature_target_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    feature_columns = select_feature_columns(train)
    missing_targets = [column for column in TARGET_COLUMNS if column not in train.columns or column not in test.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")
    if not feature_columns:
        raise ValueError("No numeric feature columns available after leakage exclusions.")

    x_train = train[feature_columns].copy()
    x_test = test[feature_columns].copy()
    y_train = train[TARGET_COLUMNS].copy()
    y_test = test[TARGET_COLUMNS].copy()
    return x_train, x_test, y_train, y_test, feature_columns


def prepare_modelling_data(
    features: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train, test = time_series_train_test_split(features, test_size=test_size)
    x_train, x_test, y_train, y_test, feature_columns = create_feature_target_matrices(train, test)

    metadata = {
        "test_size_requested": test_size,
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": len(feature_columns),
        "target_columns": TARGET_COLUMNS,
        "feature_columns": feature_columns,
        "train_start_date": train["GAME_DATE"].min().strftime("%Y-%m-%d"),
        "train_end_date": train["GAME_DATE"].max().strftime("%Y-%m-%d"),
        "test_start_date": test["GAME_DATE"].min().strftime("%Y-%m-%d"),
        "test_end_date": test["GAME_DATE"].max().strftime("%Y-%m-%d"),
    }
    return x_train, x_test, y_train, y_test, metadata


def write_modelling_data(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    metadata: dict[str, object],
    output_dir: Path = MODELLING_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train.to_parquet(output_dir / "X_train.parquet", index=False)
    x_test.to_parquet(output_dir / "X_test.parquet", index=False)
    y_train.to_parquet(output_dir / "y_train.parquet", index=False)
    y_test.to_parquet(output_dir / "y_test.parquet", index=False)
    with (output_dir / "split_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare time-aware train/test matrices for modelling.")
    parser.add_argument("--features", type=Path, default=FEATURES_FULL_PATH)
    parser.add_argument("--output-dir", type=Path, default=MODELLING_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = load_completed_features(args.features)
    x_train, x_test, y_train, y_test, metadata = prepare_modelling_data(features, test_size=args.test_size)

    print("Modelling Data Split")
    print("-------------------")
    print(f"Train rows: {metadata['train_rows']:,}")
    print(f"Test rows: {metadata['test_rows']:,}")
    print(f"Feature columns: {metadata['feature_count']:,}")
    print(f"Target columns: {', '.join(TARGET_COLUMNS)}")
    print(f"Train date range: {metadata['train_start_date']} to {metadata['train_end_date']}")
    print(f"Test date range: {metadata['test_start_date']} to {metadata['test_end_date']}")

    if not args.no_export:
        write_modelling_data(x_train, x_test, y_train, y_test, metadata, args.output_dir)
        print(f"\nSaved modelling matrices to {args.output_dir}")


if __name__ == "__main__":
    main()
