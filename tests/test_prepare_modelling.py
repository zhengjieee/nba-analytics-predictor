from __future__ import annotations

import unittest

import pandas as pd

from src.prepare_modelling import (
    TARGET_COLUMNS,
    prepare_modelling_data,
    select_feature_columns,
    time_series_train_test_split,
)


class ModellingPrepTests(unittest.TestCase):
    def make_features(self) -> pd.DataFrame:
        rows = []
        dates = [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
        ]
        for index, game_date in enumerate(dates, start=1):
            row = {
                "SEASON_ID": 22023,
                "PLAYER_ID": 100 + index,
                "PLAYER_NAME": f"Player {index}",
                "TEAM_ID": 1610612747,
                "TEAM_ABBREVIATION": "LAL",
                "TEAM_NAME": "Los Angeles Lakers",
                "GAME_ID": index,
                "GAME_DATE": game_date,
                "MATCHUP": "LAL vs. BOS",
                "WL": "W",
                "MIN": 30,
                "FGM": 5,
                "FGA": 10,
                "FG_PCT": 0.5,
                "3PM": 2,
                "FG3A": 4,
                "FG3_PCT": 0.5,
                "FTM": 4,
                "FTA": 5,
                "FT_PCT": 0.8,
                "OREB": 1,
                "DREB": 4,
                "REB": 5,
                "AST": 6,
                "STL": 1,
                "BLK": 1,
                "TOV": 2,
                "PF": 2,
                "PTS": 16,
                "PLUS_MINUS": 5,
                "FANTASY_PTS": 35.0,
                "VIDEO_AVAILABLE": 1,
                "SEASON": "2023-24",
                "PLAYER_SLUG": f"player-{index}",
                "BIRTHDATE": "2000-01-01T00:00:00",
                "SCHOOL": "NA",
                "COUNTRY": "USA",
                "HEIGHT": "6-6",
                "POSITION": "Guard",
                "ROSTERSTATUS": "Active",
                "DRAFT_YEAR": "2020",
                "DRAFT_ROUND": "1",
                "DRAFT_PICK": "10",
                "opponent_team_id": 1610612738,
                "opponent_abbreviation": "BOS",
                "days_rest_category": "1",
                "pts_rolling_5g_avg": 14.0 + index,
                "fantasy_pts_rolling_5g_avg": 30.0 + index,
                "pts_change_from_previous": 1.0,
                "fantasy_pts_change_from_previous": 2.0,
                "home_game": 1,
                "away_game": 0,
                "opponent_win_pct": 0.6,
                "opponent_def_rating": 112.0,
                "career_games_played": index - 1,
                "season_number": 4,
                "age_at_game": 24.0,
                "height_inches": 78,
                "draft_pick_numeric": 10,
                "is_undrafted": 0,
                "position_Guard": 1,
                "position_Forward": 0,
                "position_Center": 0,
                "true_shooting_pct": 0.7,
                "effective_fg_pct": 0.6,
                "three_point_attempt_rate": 0.4,
                "free_throw_attempt_rate": 0.5,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def test_time_series_split_uses_later_games_for_test_set(self) -> None:
        features = self.make_features()

        train, test = time_series_train_test_split(features, test_size=0.5)

        self.assertLess(train["GAME_DATE"].max(), test["GAME_DATE"].min())
        self.assertEqual(train["GAME_DATE"].max(), pd.Timestamp("2024-01-02"))
        self.assertEqual(test["GAME_DATE"].min(), pd.Timestamp("2024-01-03"))

    def test_feature_selection_excludes_targets_and_same_game_results(self) -> None:
        features = self.make_features()

        feature_columns = select_feature_columns(features)

        self.assertIn("pts_rolling_5g_avg", feature_columns)
        self.assertIn("home_game", feature_columns)
        self.assertIn("opponent_def_rating", feature_columns)
        self.assertNotIn("PTS", feature_columns)
        self.assertNotIn("FANTASY_PTS", feature_columns)
        self.assertNotIn("FGA", feature_columns)
        self.assertNotIn("true_shooting_pct", feature_columns)
        self.assertNotIn("PLAYER_ID", feature_columns)
        self.assertNotIn("pts_change_from_previous", feature_columns)
        self.assertNotIn("fantasy_pts_change_from_previous", feature_columns)

    def test_prepare_modelling_data_creates_x_and_y_matrices(self) -> None:
        features = self.make_features()

        x_train, x_test, y_train, y_test, metadata = prepare_modelling_data(features, test_size=0.5)

        self.assertEqual(list(y_train.columns), TARGET_COLUMNS)
        self.assertEqual(list(y_test.columns), TARGET_COLUMNS)
        self.assertEqual(metadata["target_columns"], TARGET_COLUMNS)
        self.assertEqual(metadata["train_rows"], len(x_train))
        self.assertEqual(metadata["test_rows"], len(x_test))
        self.assertEqual(set(x_train.columns), set(x_test.columns))


if __name__ == "__main__":
    unittest.main()
