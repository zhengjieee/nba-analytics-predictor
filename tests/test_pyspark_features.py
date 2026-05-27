from __future__ import annotations

import math
import sys
import unittest
from datetime import date

from pyspark.sql import SparkSession

from src.pyspark_features import add_rolling_average_std_features


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


if __name__ == "__main__":
    sys.exit(unittest.main())
