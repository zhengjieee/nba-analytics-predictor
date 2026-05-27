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


def print_rolling_feature_summary(df: DataFrame) -> None:
    rolling_columns = [
        column
        for column in df.columns
        if "_rolling_" in column and (column.endswith("_avg") or column.endswith("_std"))
    ]
    print("\nRolling Feature Checks")
    print("----------------------")
    print(f"Rolling avg/std feature columns: {len(rolling_columns)}")
    print(f"Expected rolling avg/std feature columns: {len(ROLLING_TARGETS) * len(ROLLING_WINDOWS) * 2}")
    print("Sample rolling feature columns:")
    for column in rolling_columns[:12]:
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
        featured_logs = add_rolling_average_std_features(partitioned_logs)
        print_rolling_feature_summary(featured_logs)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
