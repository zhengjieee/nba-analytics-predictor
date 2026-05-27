from __future__ import annotations

import math
import sys
import unittest
from datetime import date

from pyspark.sql import SparkSession

from src.pyspark_features import (
    add_all_time_series_features,
    add_cumulative_season_total_features,
    add_historical_min_max_features,
    add_lag_features,
    add_rolling_average_std_features,
    get_time_series_feature_columns,
)


class RollingFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spark = (
            SparkSession.builder.appName("nba-analytics-predictor-tests")
            .master("local[2]")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_rolling_features_use_prior_games_by_player(self) -> None:
        rows = [
            (1, 101, date(2024, 1, 1), 10),
            (2, 101, date(2024, 1, 2), 20),
            (3, 101, date(2024, 1, 3), 30),
            (4, 101, date(2024, 1, 4), 40),
            (5, 101, date(2024, 1, 5), 50),
            (6, 202, date(2024, 1, 1), 100),
            (7, 202, date(2024, 1, 2), 200),
        ]
        df = self.spark.createDataFrame(rows, ["GAME_ID", "PLAYER_ID", "GAME_DATE", "PTS"])

        result = add_rolling_average_std_features(df, targets=["PTS"], windows=[3])
        collected = {
            (row["PLAYER_ID"], row["GAME_ID"]): row.asDict()
            for row in result.collect()
        }

        self.assertIsNone(collected[(101, 1)]["pts_rolling_3g_avg"])
        self.assertEqual(collected[(101, 4)]["pts_rolling_3g_avg"], 20)
        self.assertTrue(math.isclose(collected[(101, 4)]["pts_rolling_3g_std"], 10))
        self.assertEqual(collected[(101, 5)]["pts_rolling_3g_avg"], 30)

        self.assertIsNone(collected[(202, 6)]["pts_rolling_3g_avg"])
        self.assertEqual(collected[(202, 7)]["pts_rolling_3g_avg"], 100)

    def test_historical_min_max_separates_season_and_career(self) -> None:
        rows = [
            (1, 101, "2023-24", date(2024, 1, 1), 10),
            (2, 101, "2023-24", date(2024, 1, 2), 30),
            (3, 101, "2024-25", date(2024, 10, 1), 20),
            (4, 101, "2024-25", date(2024, 10, 2), 40),
        ]
        df = self.spark.createDataFrame(rows, ["GAME_ID", "PLAYER_ID", "SEASON", "GAME_DATE", "PTS"])

        result = add_historical_min_max_features(df, targets=["PTS"])
        collected = {
            row["GAME_ID"]: row.asDict()
            for row in result.collect()
        }

        self.assertIsNone(collected[1]["pts_season_min"])
        self.assertIsNone(collected[1]["pts_career_min"])
        self.assertEqual(collected[2]["pts_season_min"], 10)
        self.assertEqual(collected[2]["pts_season_max"], 10)
        self.assertIsNone(collected[3]["pts_season_min"])
        self.assertEqual(collected[3]["pts_career_min"], 10)
        self.assertEqual(collected[3]["pts_career_max"], 30)
        self.assertEqual(collected[4]["pts_season_min"], 20)
        self.assertEqual(collected[4]["pts_season_max"], 20)

    def test_cumulative_season_totals_reset_each_season(self) -> None:
        rows = [
            (1, 101, "2023-24", date(2024, 1, 1), 10),
            (2, 101, "2023-24", date(2024, 1, 2), 30),
            (3, 101, "2024-25", date(2024, 10, 1), 20),
            (4, 101, "2024-25", date(2024, 10, 2), 40),
        ]
        df = self.spark.createDataFrame(rows, ["GAME_ID", "PLAYER_ID", "SEASON", "GAME_DATE", "PTS"])

        result = add_cumulative_season_total_features(df, targets=["PTS"])
        collected = {
            row["GAME_ID"]: row.asDict()
            for row in result.collect()
        }

        self.assertIsNone(collected[1]["pts_season_cumulative_total"])
        self.assertEqual(collected[2]["pts_season_cumulative_total"], 10)
        self.assertIsNone(collected[3]["pts_season_cumulative_total"])
        self.assertEqual(collected[4]["pts_season_cumulative_total"], 20)

    def test_lag_features_use_previous_career_game(self) -> None:
        rows = [
            (1, 101, date(2024, 1, 1), 10),
            (2, 101, date(2024, 1, 2), 30),
            (3, 101, date(2024, 1, 3), 20),
            (4, 202, date(2024, 1, 1), 100),
        ]
        df = self.spark.createDataFrame(rows, ["GAME_ID", "PLAYER_ID", "GAME_DATE", "PTS"])

        result = add_lag_features(df, targets=["PTS"])
        collected = {
            (row["PLAYER_ID"], row["GAME_ID"]): row.asDict()
            for row in result.collect()
        }

        self.assertIsNone(collected[(101, 1)]["pts_lag_1"])
        self.assertIsNone(collected[(101, 1)]["pts_change_from_previous"])
        self.assertEqual(collected[(101, 2)]["pts_lag_1"], 10)
        self.assertEqual(collected[(101, 2)]["pts_change_from_previous"], 20)
        self.assertEqual(collected[(101, 3)]["pts_lag_1"], 30)
        self.assertEqual(collected[(101, 3)]["pts_change_from_previous"], -10)
        self.assertIsNone(collected[(202, 4)]["pts_lag_1"])

    def test_all_time_series_feature_count(self) -> None:
        rows = [
            (1, 101, "2023-24", date(2024, 1, 1), 10, 5),
            (2, 101, "2023-24", date(2024, 1, 2), 30, 7),
        ]
        df = self.spark.createDataFrame(rows, ["GAME_ID", "PLAYER_ID", "SEASON", "GAME_DATE", "PTS", "REB"])

        result = add_all_time_series_features(df, targets=["PTS", "REB"], windows=[1, 2])
        added_columns = set(result.columns) - set(df.columns)

        self.assertEqual(len(added_columns), 22)
        self.assertEqual(len(get_time_series_feature_columns(result)), 22)


if __name__ == "__main__":
    sys.exit(unittest.main())
