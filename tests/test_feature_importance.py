from __future__ import annotations

import unittest

import pandas as pd

from src.feature_importance import classify_feature_family, summarize_feature_families


class FeatureImportanceTests(unittest.TestCase):
    def test_classify_feature_family(self) -> None:
        self.assertEqual(classify_feature_family("pts_rolling_5g_avg"), "time_series")
        self.assertEqual(classify_feature_family("reb_lag_1"), "time_series")
        self.assertEqual(classify_feature_family("opponent_win_pct"), "contextual")
        self.assertEqual(classify_feature_family("position_Guard"), "player_specific")
        self.assertEqual(classify_feature_family("some_other_numeric_feature"), "other")

    def test_summarize_feature_families_sums_gain_share_by_target(self) -> None:
        importance = pd.DataFrame(
            [
                {"target": "PTS", "feature_family": "time_series", "gain_share": 0.4},
                {"target": "PTS", "feature_family": "time_series", "gain_share": 0.3},
                {"target": "PTS", "feature_family": "contextual", "gain_share": 0.2},
                {"target": "PTS", "feature_family": "other", "gain_share": 0.1},
            ]
        )

        summary = summarize_feature_families(importance)

        time_series_share = summary.loc[
            (summary["target"] == "PTS") & (summary["feature_family"] == "time_series"),
            "gain_share",
        ].iloc[0]
        self.assertAlmostEqual(time_series_share, 0.7)


if __name__ == "__main__":
    unittest.main()
