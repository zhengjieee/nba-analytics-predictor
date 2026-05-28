from __future__ import annotations

import unittest

import pandas as pd

from src.pandas_features import (
    add_advanced_metrics,
    add_contextual_features,
    add_player_specific_features,
    height_to_inches,
    merge_player_info,
    safe_divide,
)


class PandasFeatureTests(unittest.TestCase):
    def test_merge_player_info_preserves_game_rows(self) -> None:
        features = pd.DataFrame(
            {
                "PLAYER_ID": [1, 1, 2],
                "GAME_ID": [101, 102, 201],
                "PTS": [10, 20, 30],
            }
        )
        player_info = pd.DataFrame(
            {
                "PLAYER_ID": [1, 2],
                "HEIGHT": ["6-8", "6-3"],
                "POSITION": ["Forward", "Guard"],
            }
        )

        merged = merge_player_info(features, player_info)

        self.assertEqual(len(merged), len(features))
        self.assertEqual(merged["PLAYER_ID"].nunique(), 2)
        self.assertEqual(merged.loc[merged["GAME_ID"].eq(101), "HEIGHT"].item(), "6-8")
        self.assertEqual(merged.loc[merged["GAME_ID"].eq(201), "POSITION"].item(), "Guard")

    def test_merge_player_info_rejects_duplicate_player_metadata(self) -> None:
        features = pd.DataFrame({"PLAYER_ID": [1], "GAME_ID": [101], "PTS": [10]})
        player_info = pd.DataFrame({"PLAYER_ID": [1, 1], "HEIGHT": ["6-8", "6-9"]})

        with self.assertRaises(ValueError):
            merge_player_info(features, player_info)

    def test_contextual_features_parse_matchup_and_merge_opponent_defense(self) -> None:
        features = pd.DataFrame(
            {
                "PLAYER_ID": [1, 1, 1, 1, 1],
                "GAME_ID": [101, 102, 103, 104, 105],
                "GAME_DATE": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-04", "2024-01-08"]
                ),
                "MATCHUP": ["LAL vs. BOS", "LAL @ BOS", "LAL vs. BOS", "LAL @ BOS", "LAL vs. BOS"],
                "SEASON": ["2023-24"] * 5,
                "TEAM_ID": [1610612747] * 5,
                "WL": ["W", "L", "W", "L", "W"],
                "FANTASY_PTS": [40.0, 50.0, 45.0, 42.0, 55.0],
            }
        )
        opponent_defense = pd.DataFrame(
            {
                "SEASON": ["2023-24"],
                "TEAM_ID": [1610612738],
                "W_PCT": [0.75],
                "E_DEF_RATING": [109.5],
                "DEF_RATING": [110.0],
                "DEF_RATING_RANK": [3],
                "PACE": [98.0],
                "POSS": [8000],
            }
        )

        result = add_contextual_features(features, opponent_defense)

        self.assertEqual(result.loc[0, "home_game"], 1)
        self.assertEqual(result.loc[1, "away_game"], 1)
        self.assertEqual(result.loc[0, "opponent_abbreviation"], "BOS")
        self.assertEqual(result.loc[0, "opponent_def_rating"], 110.0)
        self.assertEqual(result.loc[0, "opponent_win_pct"], 0.75)
        self.assertEqual(result.loc[0, "days_rest_category"], "first_game")
        self.assertEqual(result.loc[1, "days_rest_category"], "1")
        self.assertEqual(result.loc[2, "days_rest_category"], "2")
        self.assertEqual(result.loc[3, "days_rest_category"], "0")
        self.assertEqual(result.loc[4, "days_rest_category"], "3+")

    def test_player_specific_features_create_model_ready_values(self) -> None:
        features = pd.DataFrame(
            {
                "PLAYER_ID": [1, 1],
                "GAME_ID": [101, 102],
                "GAME_DATE": pd.to_datetime(["2024-01-01", "2024-01-03"]),
                "SEASON": ["2023-24", "2023-24"],
                "BIRTHDATE": ["2000-01-01T00:00:00", "2000-01-01T00:00:00"],
                "HEIGHT": ["6-8", "6-8"],
                "WEIGHT": [220, 220],
                "POSITION": ["Forward-Center", "Forward-Center"],
                "FROM_YEAR": [2020, 2020],
                "DRAFT_PICK": ["Undrafted", "Undrafted"],
            }
        )

        result = add_player_specific_features(features)

        self.assertEqual(result.loc[0, "career_games_played"], 0)
        self.assertEqual(result.loc[1, "career_games_played"], 1)
        self.assertEqual(result.loc[0, "season_number"], 4)
        self.assertEqual(result.loc[0, "height_inches"], 80)
        self.assertEqual(result.loc[0, "draft_pick_numeric"], 61)
        self.assertEqual(result.loc[0, "is_undrafted"], 1)
        self.assertEqual(result.loc[0, "position_Forward"], 1)
        self.assertEqual(result.loc[0, "position_Center"], 1)

    def test_advanced_metrics_use_safe_denominators(self) -> None:
        features = pd.DataFrame(
            {
                "PTS": [20, 0],
                "FGA": [10, 0],
                "FGM": [5, 0],
                "3PM": [2, 0],
                "FG3A": [4, 0],
                "FTA": [5, 0],
                "FTM": [4, 0],
                "REB": [8, 0],
                "AST": [6, 0],
                "STL": [1, 0],
                "BLK": [1, 0],
                "TOV": [2, 0],
                "MIN": [30, 0],
            }
        )

        result = add_advanced_metrics(features)

        self.assertAlmostEqual(result.loc[0, "true_shooting_pct"], 20 / (2 * (10 + 0.44 * 5)))
        self.assertAlmostEqual(result.loc[0, "effective_fg_pct"], (5 + 0.5 * 2) / 10)
        self.assertAlmostEqual(result.loc[0, "three_point_attempt_rate"], 0.4)
        self.assertTrue(pd.isna(result.loc[1, "true_shooting_pct"]))

    def test_small_helpers(self) -> None:
        self.assertEqual(height_to_inches("6-9"), 81)
        result = safe_divide(pd.Series([1, 1]), pd.Series([2, 0]))
        self.assertEqual(result.iloc[0], 0.5)
        self.assertTrue(pd.isna(result.iloc[1]))


if __name__ == "__main__":
    unittest.main()
