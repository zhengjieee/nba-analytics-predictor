from __future__ import annotations

import unittest

import pandas as pd

from src.pandas_features import merge_player_info


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


if __name__ == "__main__":
    unittest.main()
