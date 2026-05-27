"""
Complete Pandas-based feature enrichment after PySpark processing.

Loads the PySpark-generated player-game feature table, 
and merges one-row-per-player metadata from player_info.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PYSPARK_FEATURES_PATH = PROCESSED_DIR / "features_pyspark.parquet"
PLAYER_INFO_PATH = RAW_DIR / "player_info.csv"
PLAYER_INFO_FEATURES_PATH = PROCESSED_DIR / "features_player_info.parquet"

PLAYER_INFO_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_SLUG",
    "BIRTHDATE",
    "SCHOOL",
    "COUNTRY",
    "HEIGHT",
    "WEIGHT",
    "SEASON_EXP",
    "POSITION",
    "ROSTERSTATUS",
    "FROM_YEAR",
    "TO_YEAR",
    "DRAFT_YEAR",
    "DRAFT_ROUND",
    "DRAFT_PICK",
]


def load_pyspark_features(path: Path = PYSPARK_FEATURES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_player_info(path: Path = PLAYER_INFO_PATH) -> pd.DataFrame:
    player_info = pd.read_csv(path)
    return player_info[PLAYER_INFO_COLUMNS].copy()


def merge_player_info(features: pd.DataFrame, player_info: pd.DataFrame) -> pd.DataFrame:
    """Merge one-row-per-player metadata onto player-game features."""
    if player_info["PLAYER_ID"].duplicated().any():
        duplicated = player_info.loc[player_info["PLAYER_ID"].duplicated(), "PLAYER_ID"].tolist()
        raise ValueError(f"player_info contains duplicate PLAYER_ID values: {duplicated[:5]}")

    merged = features.merge(
        player_info,
        on="PLAYER_ID",
        how="left",
        validate="many_to_one",
    )
    return merged


def validate_player_info_merge(features: pd.DataFrame, player_info: pd.DataFrame, merged: pd.DataFrame) -> None:
    key_columns = ["PLAYER_ID", "GAME_ID"]
    metadata_columns = [column for column in PLAYER_INFO_COLUMNS if column != "PLAYER_ID"]

    missing_players = set(features["PLAYER_ID"].unique()) - set(player_info["PLAYER_ID"].unique())
    duplicate_keys = merged.duplicated(key_columns).sum()
    missing_metadata = merged[metadata_columns].isna().sum()
    missing_metadata = missing_metadata[missing_metadata > 0]

    print("Player Info Merge Validation")
    print("----------------------------")
    print(f"Input feature rows: {len(features):,}")
    print(f"Merged rows: {len(merged):,}")
    print(f"Input feature columns: {len(features.columns):,}")
    print(f"Merged columns: {len(merged.columns):,}")
    print(f"Player info rows: {len(player_info):,}")
    print(f"Unique players in features: {features['PLAYER_ID'].nunique():,}")
    print(f"Unique players in player_info: {player_info['PLAYER_ID'].nunique():,}")
    print(f"Missing players after merge: {len(missing_players):,}")
    print(f"Duplicate PLAYER_ID + GAME_ID rows after merge: {duplicate_keys:,}")

    if len(features) != len(merged):
        raise ValueError("Merge changed row count.")
    if missing_players:
        raise ValueError(f"Missing player metadata for PLAYER_ID values: {sorted(missing_players)[:10]}")
    if duplicate_keys:
        raise ValueError("Merge introduced duplicate PLAYER_ID + GAME_ID rows.")

    print("\nMetadata missing-value counts:")
    if len(missing_metadata):
        print(missing_metadata.to_string())
    else:
        print("None")


def write_processed_features(df: pd.DataFrame, path: Path = PLAYER_INFO_FEATURES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"\nSaved merged player-info features to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge PySpark features with player metadata using Pandas.")
    parser.add_argument("--features", type=Path, default=PYSPARK_FEATURES_PATH)
    parser.add_argument("--player-info", type=Path, default=PLAYER_INFO_PATH)
    parser.add_argument("--output", type=Path, default=PLAYER_INFO_FEATURES_PATH)
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = load_pyspark_features(args.features)
    player_info = load_player_info(args.player_info)
    merged = merge_player_info(features, player_info)
    validate_player_info_merge(features, player_info, merged)

    if not args.no_export:
        write_processed_features(merged, args.output)


if __name__ == "__main__":
    main()
