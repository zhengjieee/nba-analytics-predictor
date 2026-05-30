"""
Completes the Pandas stage of the feature pipeline.

The script first merges one-row-per-player metadata from player_info.csv onto
the PySpark-generated player-game features, producing features_player_info.parquet.

It builds the final modeling table by adding contextual, player-specific, and advanced features, 
then writes features_full.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PYSPARK_FEATURES_PATH = PROCESSED_DIR / "features_pyspark.parquet"
PLAYER_INFO_PATH = RAW_DIR / "player_info.csv"
OPPONENT_DEFENSE_PATH = RAW_DIR / "opponent_defense.csv"
PLAYER_INFO_FEATURES_PATH = PROCESSED_DIR / "features_player_info.parquet"
FEATURES_FULL_PATH = PROCESSED_DIR / "features_full.parquet"

TEAM_ABBREVIATION_TO_ID = {
    "ATL": 1610612737,
    "BOS": 1610612738,
    "CLE": 1610612739,
    "NOP": 1610612740,
    "NOH": 1610612740,
    "NOK": 1610612740,
    "CHI": 1610612741,
    "DAL": 1610612742,
    "DEN": 1610612743,
    "GSW": 1610612744,
    "HOU": 1610612745,
    "LAC": 1610612746,
    "LAL": 1610612747,
    "MIA": 1610612748,
    "MIL": 1610612749,
    "MIN": 1610612750,
    "BKN": 1610612751,
    "NJN": 1610612751,
    "NYK": 1610612752,
    "ORL": 1610612753,
    "IND": 1610612754,
    "PHI": 1610612755,
    "PHX": 1610612756,
    "POR": 1610612757,
    "SAC": 1610612758,
    "SAS": 1610612759,
    "OKC": 1610612760,
    "SEA": 1610612760,
    "TOR": 1610612761,
    "UTA": 1610612762,
    "MEM": 1610612763,
    "WAS": 1610612764,
    "DET": 1610612765,
    "CHA": 1610612766,
}

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


def load_opponent_defense(path: Path = OPPONENT_DEFENSE_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


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


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def height_to_inches(height: str | float) -> float:
    if pd.isna(height):
        return np.nan
    feet, inches = str(height).split("-")
    return int(feet) * 12 + int(inches)


def season_start_year(season: str) -> int:
    return int(str(season).split("-")[0])


def parse_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["home_game"] = featured["MATCHUP"].str.contains("vs.", regex=False).astype(int)
    featured["away_game"] = featured["MATCHUP"].str.contains("@", regex=False).astype(int)
    featured["opponent_abbreviation"] = featured["MATCHUP"].str.extract(r"(?:@|vs\.)\s+([A-Z]{2,3})")[0]
    featured["opponent_team_id"] = featured["opponent_abbreviation"].map(TEAM_ABBREVIATION_TO_ID)
    return featured


def add_days_rest(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).copy()
    featured["days_rest"] = featured.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days
    featured["days_rest_category"] = np.select(
        [
            featured["days_rest"].isna(),
            featured["days_rest"].eq(0),
            featured["days_rest"].eq(1),
            featured["days_rest"].eq(2),
            featured["days_rest"] >= 3,
        ],
        ["first_game", "0", "1", "2", "3+"],
        default="unknown",
    )
    return featured.sort_index()


def merge_opponent_defense(features: pd.DataFrame, opponent_defense: pd.DataFrame) -> pd.DataFrame:
    defense_columns = [
        "SEASON",
        "TEAM_ID",
        "W_PCT",
        "DEF_RATING",
    ]
    defense = opponent_defense[defense_columns].rename(
        columns={
            "TEAM_ID": "opponent_team_id",
            "W_PCT": "opponent_win_pct",
            "DEF_RATING": "opponent_def_rating",
        }
    )
    return features.merge(
        defense,
        on=["SEASON", "opponent_team_id"],
        how="left",
        validate="many_to_one",
    )


def add_opponent_recent_form(features: pd.DataFrame) -> pd.DataFrame:
    team_games = (
        features[["GAME_ID", "TEAM_ID", "GAME_DATE", "WL"]]
        .drop_duplicates(["GAME_ID", "TEAM_ID"])
        .sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"])
        .copy()
    )
    team_games["team_win"] = team_games["WL"].eq("W").astype(int)
    team_games["opponent_recent_form"] = (
        team_games.groupby("TEAM_ID")["team_win"]
        .transform(lambda values: values.shift(1).rolling(window=5, min_periods=1).mean())
    )
    recent_form = team_games.rename(columns={"TEAM_ID": "opponent_team_id"})[
        ["GAME_ID", "opponent_team_id", "opponent_recent_form"]
    ]
    return features.merge(
        recent_form,
        on=["GAME_ID", "opponent_team_id"],
        how="left",
        validate="many_to_one",
    )


def add_contextual_features(features: pd.DataFrame, opponent_defense: pd.DataFrame) -> pd.DataFrame:
    featured = features.copy()
    featured["GAME_DATE"] = pd.to_datetime(featured["GAME_DATE"])
    featured = parse_matchup_features(featured)
    featured = add_days_rest(featured)
    featured = merge_opponent_defense(featured, opponent_defense)
    return add_opponent_recent_form(featured)


def add_player_specific_features(features: pd.DataFrame) -> pd.DataFrame:
    featured = features.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).copy()
    featured["career_games_played"] = featured.groupby("PLAYER_ID").cumcount()
    featured["season_start_year"] = featured["SEASON"].map(season_start_year)
    featured["season_number"] = featured["season_start_year"] - featured["FROM_YEAR"] + 1
    featured["BIRTHDATE"] = pd.to_datetime(featured["BIRTHDATE"])
    featured["age_at_game"] = (featured["GAME_DATE"] - featured["BIRTHDATE"]).dt.days / 365.25
    featured["height_inches"] = featured["HEIGHT"].map(height_to_inches)
    featured["draft_pick_numeric"] = pd.to_numeric(featured["DRAFT_PICK"], errors="coerce")
    featured["is_undrafted"] = featured["DRAFT_PICK"].astype(str).str.lower().eq("undrafted").astype(int)
    featured["draft_pick_numeric"] = featured["draft_pick_numeric"].fillna(61)

    position_dummies = featured["POSITION"].str.get_dummies(sep="-").add_prefix("position_")
    return pd.concat([featured, position_dummies], axis=1).sort_index()


def add_advanced_metrics(features: pd.DataFrame) -> pd.DataFrame:
    featured = features.copy()
    featured["true_shooting_pct"] = safe_divide(
        featured["PTS"],
        2 * (featured["FGA"] + 0.44 * featured["FTA"]),
    )
    featured["effective_fg_pct"] = safe_divide(
        featured["FGM"] + 0.5 * featured["3PM"],
        featured["FGA"],
    )
    featured["three_point_attempt_rate"] = safe_divide(featured["FG3A"], featured["FGA"])
    featured["free_throw_attempt_rate"] = safe_divide(featured["FTA"], featured["FGA"])
    return featured


def build_full_features(features: pd.DataFrame, opponent_defense: pd.DataFrame) -> pd.DataFrame:
    featured = add_contextual_features(features, opponent_defense)
    featured = add_player_specific_features(featured)
    return add_advanced_metrics(featured)


def validate_full_features(base: pd.DataFrame, full: pd.DataFrame) -> None:
    key_columns = ["PLAYER_ID", "GAME_ID"]
    important_columns = [
        "home_game",
        "opponent_abbreviation",
        "opponent_def_rating",
        "age_at_game",
        "height_inches",
        "draft_pick_numeric",
        "true_shooting_pct",
        "effective_fg_pct",
    ]
    missing_counts = full[important_columns].isna().sum()
    duplicate_keys = full.duplicated(key_columns).sum()

    print("Full Feature Validation")
    print("-----------------------")
    print(f"Base rows: {len(base):,}")
    print(f"Full rows: {len(full):,}")
    print(f"Base columns: {len(base.columns):,}")
    print(f"Full columns: {len(full.columns):,}")
    print(f"New columns: {len(full.columns) - len(base.columns):,}")
    print(f"Duplicate PLAYER_ID + GAME_ID rows: {duplicate_keys:,}")
    print("\nImportant missing-value counts:")
    print(missing_counts.to_string())

    if len(base) != len(full):
        raise ValueError("Full feature creation changed row count.")
    if duplicate_keys:
        raise ValueError("Full feature creation introduced duplicate PLAYER_ID + GAME_ID rows.")
    if full["opponent_team_id"].isna().any():
        missing = sorted(full.loc[full["opponent_team_id"].isna(), "opponent_abbreviation"].dropna().unique())
        raise ValueError(f"Missing opponent team ID mappings for: {missing}")


def write_processed_features(df: pd.DataFrame, path: Path = PLAYER_INFO_FEATURES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"\nSaved merged player-info features to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge PySpark features with player metadata using Pandas.")
    parser.add_argument("--features", type=Path, default=PYSPARK_FEATURES_PATH)
    parser.add_argument("--player-info", type=Path, default=PLAYER_INFO_PATH)
    parser.add_argument("--opponent-defense", type=Path, default=OPPONENT_DEFENSE_PATH)
    parser.add_argument("--output", type=Path, default=PLAYER_INFO_FEATURES_PATH)
    parser.add_argument("--full-output", type=Path, default=FEATURES_FULL_PATH)
    parser.add_argument("--full", action="store_true", help="Build contextual, player-specific, and advanced features.")
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = load_pyspark_features(args.features)
    player_info = load_player_info(args.player_info)
    metadata_columns = [column for column in PLAYER_INFO_COLUMNS if column != "PLAYER_ID"]
    if all(column in features.columns for column in metadata_columns):
        merged = features
        print("Player metadata already present; skipping player_info merge.")
    else:
        merged = merge_player_info(features, player_info)
        validate_player_info_merge(features, player_info, merged)

    if args.full:
        opponent_defense = load_opponent_defense(args.opponent_defense)
        full = build_full_features(merged, opponent_defense)
        validate_full_features(merged, full)
        if not args.no_export:
            write_processed_features(full, args.full_output)
        return

    if not args.no_export:
        write_processed_features(merged, args.output)


if __name__ == "__main__":
    main()
