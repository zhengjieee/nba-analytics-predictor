"""
Analyse feature importance from trained tree-based models.

This script reads the saved models for all eight targets, extracts gain-based
importance scores, groups features into broad families, and examines whether
time-series features are driving the models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_modelling import MODELLING_DIR, SPLIT_DIR, TARGET_COLUMNS
from src.train_models import DEFAULT_MODEL_NAME, MODEL_CONFIGS, ModelConfig, get_model_config, model_path_for_target, PRIMARY_TARGET

DEFAULT_IMPORTANCE_DIR = MODELLING_DIR / DEFAULT_MODEL_NAME / "feature_importance"
IMPORTANCE_OUTPUT_PATH = DEFAULT_IMPORTANCE_DIR / "importance_scores.json"
SUMMARY_OUTPUT_PATH = DEFAULT_IMPORTANCE_DIR / "importance_summary.md"

TIME_SERIES_MARKERS = (
    "_rolling_",
    "_lag_",
    "_season_",
    "_career_",
    "_cumulative_",
)

CONTEXTUAL_FEATURES = {
    "home_game",
    "away_game",
    "days_rest",
    "opponent_win_pct",
    "opponent_def_rating",
    "opponent_recent_form",
}

PLAYER_SPECIFIC_FEATURES = {
    "career_games_played",
    "season_start_year",
    "season_number",
    "age_at_game",
    "height_inches",
    "draft_pick_numeric",
    "is_undrafted",
    "WEIGHT",
    "SEASON_EXP",
    "FROM_YEAR",
    "TO_YEAR",
}


def classify_feature_family(feature: str) -> str:
    if any(marker in feature for marker in TIME_SERIES_MARKERS):
        return "time_series"
    if feature in CONTEXTUAL_FEATURES or feature.startswith("opponent_"):
        return "contextual"
    if feature in PLAYER_SPECIFIC_FEATURES or feature.startswith("position_"):
        return "player_specific"
    return "other"


def importance_dir_for_model(config: ModelConfig) -> Path:
    return MODELLING_DIR / config.name / "feature_importance"


def load_feature_columns(split_metadata_path: Path = SPLIT_DIR / "split_metadata.json") -> list[str]:
    with split_metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return list(metadata["feature_columns"])


def load_gain_scores(model_path: Path, config: ModelConfig) -> dict[str, float]:
    if config.name == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor()
        model.load_model(model_path)
        return {feature: float(score) for feature, score in model.get_booster().get_score(importance_type="gain").items()}

    if config.name == "lightgbm":
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(model_path))
        names = booster.feature_name()
        gains = booster.feature_importance(importance_type="gain")
        return {feature: float(gain) for feature, gain in zip(names, gains)}

    raise ValueError(f"Unsupported model for feature importance: {config.name}")


def build_feature_importance_table(
    feature_columns: list[str],
    targets: list[str] = TARGET_COLUMNS,
    config: ModelConfig | None = None,
    models_dir: Path | None = None,
) -> pd.DataFrame:
    model_config = MODEL_CONFIGS[DEFAULT_MODEL_NAME] if config is None else config
    rows = []
    for target in targets:
        model_path = model_path_for_target(target, model_config, models_dir=models_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model for {target}: {model_path}")

        gain_scores = load_gain_scores(model_path, model_config)
        total_gain = sum(gain_scores.values())

        for feature in feature_columns:
            gain = gain_scores.get(feature, 0.0)
            rows.append(
                {
                    "model": model_config.name,
                    "target": target,
                    "feature": feature,
                    "feature_family": classify_feature_family(feature),
                    "gain": gain,
                    "gain_share": 0.0 if total_gain == 0 else gain / total_gain,
                }
            )

    importance = pd.DataFrame(rows)
    importance["rank"] = importance.groupby("target")["gain"].rank(method="first", ascending=False).astype(int)
    return importance.sort_values(["target", "rank"]).reset_index(drop=True)


def summarize_feature_families(importance: pd.DataFrame) -> pd.DataFrame:
    summary = (
        importance.groupby(["model", "target", "feature_family"], as_index=False)["gain_share"]
        .sum()
        .sort_values(["target", "gain_share"], ascending=[True, False])
    )
    return summary


def write_feature_importance_json(
    importance: pd.DataFrame,
    family_summary: pd.DataFrame,
    output_path: Path = IMPORTANCE_OUTPUT_PATH,
) -> None:
    # This highlights the most important inputs for the project's primary target.
    # The full feature ranking for every target is still kept in all_importance.
    output = {
        "importance_type": "gain",
        "model": str(importance["model"].iloc[0]) if not importance.empty else None,
        "top_20_fantasy_points": importance[
            (importance["target"] == PRIMARY_TARGET) & (importance["rank"] <= 20)
        ].to_dict(orient="records"),
        "family_summary": family_summary.to_dict(orient="records"),
        "all_importance": importance.to_dict(orient="records"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def model_label(config: ModelConfig) -> str:
    return "XGBoost" if config.name == "xgboost" else "LightGBM"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_summary_markdown(importance: pd.DataFrame, family_summary: pd.DataFrame, config: ModelConfig) -> str:
    fantasy_top_20 = importance[(importance["target"] == PRIMARY_TARGET) & (importance["rank"] <= 20)]
    family_pivot = (
        family_summary.pivot(index="target", columns="feature_family", values="gain_share")
        .fillna(0.0)
        .reset_index()
    )

    time_series_share = family_pivot[["target", "time_series"]].sort_values("time_series", ascending=False)
    targets_with_time_series_majority = int((time_series_share["time_series"] > 0.5).sum())
    strongest_time_series_target = time_series_share.iloc[0]
    weakest_time_series_target = time_series_share.iloc[-1]

    top_rows = [
        [
            int(row.rank),
            row.feature,
            row.feature_family,
            format_pct(row.gain_share),
        ]
        for row in fantasy_top_20.itertuples(index=False)
    ]

    family_rows = []
    for row in family_pivot.itertuples(index=False):
        family_rows.append(
            [
                row.target,
                format_pct(getattr(row, "time_series", 0.0)),
                format_pct(getattr(row, "contextual", 0.0)),
                format_pct(getattr(row, "player_specific", 0.0)),
                format_pct(getattr(row, "other", 0.0)),
            ]
        )

    label = model_label(config)
    lines = [
        f"# {label} Feature Importance Summary",
        "",
        f"This summary uses {label} gain importance from the saved default models. Gain measures how much a feature improves tree splits when it is used, so larger values mean the model relied on that feature more.",
        "",
        "## Top 20 Fantasy Points Features",
        "",
        markdown_table(["Rank", "Feature", "Feature Family", "Gain Share"], top_rows),
        "",
        "## Feature Family Share By Target",
        "",
        markdown_table(
            ["Target", "Time-Series", "Contextual", "Player-Specific", "Other"],
            family_rows,
        ),
        "",
        "## Key Insights",
        "",
        f"Time-series features are the largest feature family for {targets_with_time_series_majority} of the {len(TARGET_COLUMNS)} targets. This supports keeping the PySpark feature engineering stage because the rolling, lag, season, and career-history features are doing much of the predictive work.",
        "",
        f"The strongest time-series reliance is for `{strongest_time_series_target.target}` at {format_pct(strongest_time_series_target.time_series)} of gain importance. The weakest is `{weakest_time_series_target.target}` at {format_pct(weakest_time_series_target.time_series)}, which suggests that some targets depend more on context, player profile, or noisy game-level variation.",
        "",
        "Fantasy points is the main target for this project, so its feature importance is the most useful one to review. If its top features are mostly recent-form or history features, that validates the project design: the model is learning from a player's recent and longer-term performance rather than from same-game leakage columns.",
        "",
        "Importance scores should be treated as model diagnostics, not causal explanations. Correlated rolling features can share importance unevenly, so the ranking is best read as a directional view of what the model found useful.",
    ]
    return "\n".join(lines) + "\n"


def write_summary_markdown(
    importance: pd.DataFrame,
    family_summary: pd.DataFrame,
    config: ModelConfig,
    output_path: Path = SUMMARY_OUTPUT_PATH,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_summary_markdown(importance, family_summary, config), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze feature importance from saved tree-based models.")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default=DEFAULT_MODEL_NAME)
    parser.add_argument("--split-metadata", type=Path, default=SPLIT_DIR / "split_metadata.json")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_model_config(args.model)
    models_dir = config.models_dir if args.models_dir is None else args.models_dir
    importance_dir = importance_dir_for_model(config)
    output_json = importance_dir / "importance_scores.json" if args.output_json is None else args.output_json
    output_summary = importance_dir / "importance_summary.md" if args.output_summary is None else args.output_summary

    feature_columns = load_feature_columns(args.split_metadata)
    importance = build_feature_importance_table(feature_columns, config=config, models_dir=models_dir)
    family_summary = summarize_feature_families(importance)

    print(f"{model_label(config)} Feature Importance")
    print("--------------------------")
    print("Top 20 fantasy-points features:")
    print(
        importance[(importance["target"] == PRIMARY_TARGET) & (importance["rank"] <= 20)][
            ["rank", "feature", "feature_family", "gain_share"]
        ]
        .assign(gain_share=lambda df: df["gain_share"].map(format_pct))
        .to_string(index=False)
    )

    print("\nFeature family gain share by target:")
    print(
        family_summary.pivot(index="target", columns="feature_family", values="gain_share")
        .fillna(0.0)
        .map(format_pct)
        .to_string()
    )

    if not args.no_export:
        write_feature_importance_json(importance, family_summary, output_json)
        write_summary_markdown(importance, family_summary, config, output_summary)
        print(f"\nSaved feature importance to {output_json}")
        print(f"Saved feature importance summary to {output_summary}")


if __name__ == "__main__":
    main()
