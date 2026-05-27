"""
PySpark feature engineering setup for NBA player game logs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DateType, DoubleType, IntegerType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_LOGS_PATH = PROJECT_ROOT / "data" / "raw" / "player_game_logs.csv"
ROLLING_TARGETS = ["FANTASY_PTS", "PTS", "REB", "AST", "STL", "BLK", "TOV", "3PM"]
ROLLING_WINDOWS = [5, 10, 20]

INTEGER_COLUMNS = [
    "SEASON_ID",
    "PLAYER_ID",
    "TEAM_ID",
    "GAME_ID",
    "MIN",
    "FGM",
    "FGA",
    "3PM",
    "FG3A",
    "FTM",
    "FTA",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
    "PLUS_MINUS",
    "VIDEO_AVAILABLE",
]

DOUBLE_COLUMNS = [
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "FANTASY_PTS",
]


def create_spark_session(app_name: str = "nba-analytics-predictor") -> SparkSession:
    """Create a local PySpark session for feature engineering."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    return (
        SparkSession.builder.appName(app_name)
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def load_player_game_logs(spark: SparkSession, path: Path = RAW_LOGS_PATH) -> DataFrame:
    """Load raw player game logs and apply explicit types used downstream."""
    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "false")
        .option("nullValue", "NA")
        .csv(str(path))
    )

    for column in INTEGER_COLUMNS:
        df = df.withColumn(column, F.col(column).cast(IntegerType()))

    for column in DOUBLE_COLUMNS:
        df = df.withColumn(column, F.col(column).cast(DoubleType()))

    return df.withColumn("GAME_DATE", F.to_date(F.col("GAME_DATE")).cast(DateType()))


def print_schema_summary(df: DataFrame) -> None:
    print("Schema")
    print("------")
    df.printSchema()

    row_count = df.count()
    player_count = df.select("PLAYER_ID").distinct().count()
    season_count = df.select("SEASON").distinct().count()
    date_range = df.agg(F.min("GAME_DATE").alias("min_date"), F.max("GAME_DATE").alias("max_date")).collect()[0]
    duplicate_count = (
        df.groupBy("PLAYER_ID", "GAME_ID")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    print("\nLoad Checks")
    print("-----------")
    print(f"Rows: {row_count:,}")
    print(f"Players: {player_count:,}")
    print(f"Seasons: {season_count:,}")
    print(f"Date range: {date_range['min_date']} to {date_range['max_date']}")
    print(f"Duplicate PLAYER_ID + GAME_ID groups: {duplicate_count:,}")


def verify_player_partitioning(df: DataFrame, partitions: int = 8) -> DataFrame:
    """Repartition by player and print distribution diagnostics."""
    partitioned = df.repartition(partitions, "PLAYER_ID")
    partition_counts = (
        partitioned.withColumn("spark_partition_id", F.spark_partition_id())
        .groupBy("spark_partition_id")
        .agg(
            F.count("*").alias("rows"),
            F.countDistinct("PLAYER_ID").alias("players"),
        )
        .orderBy("spark_partition_id")
    )

    print("\nPartitioning By PLAYER_ID")
    print("-------------------------")
    print(f"Requested partitions: {partitions}")
    print(f"Actual partitions: {partitioned.rdd.getNumPartitions()}")
    partition_counts.show(partitions, truncate=False)
    return partitioned


def rolling_feature_name(target: str, window_size: int, statistic: str) -> str:
    normalized_target = target.lower()
    return f"{normalized_target}_rolling_{window_size}g_{statistic}"


def historical_feature_name(target: str, scope: str, statistic: str) -> str:
    normalized_target = target.lower()
    return f"{normalized_target}_{scope}_{statistic}"


def cumulative_feature_name(target: str) -> str:
    normalized_target = target.lower()
    return f"{normalized_target}_season_cumulative_total"


def lag_feature_name(target: str) -> str:
    normalized_target = target.lower()
    return f"{normalized_target}_lag_1"


def change_feature_name(target: str) -> str:
    normalized_target = target.lower()
    return f"{normalized_target}_change_from_previous"


def add_rolling_average_std_features(
    df: DataFrame,
    targets: list[str] | None = None,
    windows: list[int] | None = None,
) -> DataFrame:
    """Add leak-safe rolling average and standard deviation features.

    Each rolling window is partitioned by player and ordered by game date. The
    current row is excluded from the window so features only use games that
    happened before the game being predicted.
    """
    targets = targets or ROLLING_TARGETS
    windows = windows or ROLLING_WINDOWS
    ordered = Window.partitionBy("PLAYER_ID").orderBy("GAME_DATE", "GAME_ID")

    featured = df
    for target in targets:
        for window_size in windows:
            prior_games = ordered.rowsBetween(-window_size, -1)
            featured = featured.withColumn(
                rolling_feature_name(target, window_size, "avg"),
                F.avg(F.col(target)).over(prior_games),
            )
            featured = featured.withColumn(
                rolling_feature_name(target, window_size, "std"),
                F.stddev_samp(F.col(target)).over(prior_games),
            )

    return featured


def add_historical_min_max_features(
    df: DataFrame,
    targets: list[str] | None = None,
) -> DataFrame:
    """Add prior season and prior career min/max features for each target."""
    targets = targets or ROLLING_TARGETS
    season_ordered = Window.partitionBy("PLAYER_ID", "SEASON").orderBy("GAME_DATE", "GAME_ID")
    career_ordered = Window.partitionBy("PLAYER_ID").orderBy("GAME_DATE", "GAME_ID")
    prior_season_games = season_ordered.rowsBetween(Window.unboundedPreceding, -1)
    prior_career_games = career_ordered.rowsBetween(Window.unboundedPreceding, -1)

    featured = df
    for target in targets:
        featured = featured.withColumn(
            historical_feature_name(target, "season", "min"),
            F.min(F.col(target)).over(prior_season_games),
        )
        featured = featured.withColumn(
            historical_feature_name(target, "season", "max"),
            F.max(F.col(target)).over(prior_season_games),
        )
        featured = featured.withColumn(
            historical_feature_name(target, "career", "min"),
            F.min(F.col(target)).over(prior_career_games),
        )
        featured = featured.withColumn(
            historical_feature_name(target, "career", "max"),
            F.max(F.col(target)).over(prior_career_games),
        )

    return featured


def add_cumulative_season_total_features(
    df: DataFrame,
    targets: list[str] | None = None,
) -> DataFrame:
    """Add prior cumulative season totals for each target."""
    targets = targets or ROLLING_TARGETS
    ordered = Window.partitionBy("PLAYER_ID", "SEASON").orderBy("GAME_DATE", "GAME_ID")
    prior_season_games = ordered.rowsBetween(Window.unboundedPreceding, -1)

    featured = df
    for target in targets:
        featured = featured.withColumn(
            cumulative_feature_name(target),
            F.sum(F.col(target)).over(prior_season_games),
        )

    return featured


def add_lag_features(
    df: DataFrame,
    targets: list[str] | None = None,
) -> DataFrame:
    """Add lag-1 and current-minus-previous features for each target."""
    targets = targets or ROLLING_TARGETS
    ordered = Window.partitionBy("PLAYER_ID").orderBy("GAME_DATE", "GAME_ID")

    featured = df
    for target in targets:
        lag_column = lag_feature_name(target)
        featured = featured.withColumn(lag_column, F.lag(F.col(target), 1).over(ordered))
        featured = featured.withColumn(change_feature_name(target), F.col(target) - F.col(lag_column))

    return featured


def add_all_time_series_features(
    df: DataFrame,
    targets: list[str] | None = None,
    windows: list[int] | None = None,
) -> DataFrame:
    """Add all planned PySpark time-series features."""
    targets = targets or ROLLING_TARGETS
    windows = windows or ROLLING_WINDOWS

    featured = add_rolling_average_std_features(df, targets=targets, windows=windows)
    featured = add_historical_min_max_features(featured, targets=targets)
    featured = add_cumulative_season_total_features(featured, targets=targets)
    return add_lag_features(featured, targets=targets)


def print_time_series_feature_summary(df: DataFrame) -> None:
    rolling_columns = [
        column
        for column in df.columns
        if "_rolling_" in column and (column.endswith("_avg") or column.endswith("_std"))
    ]
    min_max_columns = [
        column
        for column in df.columns
        if (("_season_" in column or "_career_" in column) and (column.endswith("_min") or column.endswith("_max")))
    ]
    cumulative_columns = [column for column in df.columns if column.endswith("_season_cumulative_total")]
    lag_columns = [column for column in df.columns if column.endswith("_lag_1")]
    change_columns = [column for column in df.columns if column.endswith("_change_from_previous")]

    print("\nTime-Series Feature Checks")
    print("--------------------------")
    print(f"Rolling avg/std feature columns: {len(rolling_columns)}")
    print(f"Season/career min/max feature columns: {len(min_max_columns)}")
    print(f"Cumulative season total feature columns: {len(cumulative_columns)}")
    print(f"Lag-1 feature columns: {len(lag_columns)}")
    print(f"Change-from-previous feature columns: {len(change_columns)}")
    print(f"Total planned PySpark feature columns: {len(rolling_columns) + len(min_max_columns) + len(cumulative_columns) + len(lag_columns) + len(change_columns)}")
    print(f"Total DataFrame columns: {len(df.columns)}")
    print("Sample time-series feature columns:")
    for column in (rolling_columns + min_max_columns + cumulative_columns + lag_columns + change_columns)[:12]:
        print(f"- {column}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and validate NBA player game logs with PySpark.")
    parser.add_argument("--input", type=Path, default=RAW_LOGS_PATH)
    parser.add_argument("--partitions", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    try:
        logs = load_player_game_logs(spark, args.input)
        print_schema_summary(logs)
        partitioned_logs = verify_player_partitioning(logs, partitions=args.partitions)
        featured_logs = add_all_time_series_features(partitioned_logs)
        print_time_series_feature_summary(featured_logs)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
