# Feature Importance Summary

This summary uses XGBoost gain importance from the saved default models. Gain measures how much a feature improves tree splits when it is used, so larger values mean the model relied on that feature more.

## Top 20 Fantasy Points Features

| Rank | Feature | Feature Family | Gain Share |
| --- | --- | --- | --- |
| 1 | fantasy_pts_rolling_20g_avg | time_series | 50.9% |
| 2 | fantasy_pts_rolling_10g_avg | time_series | 16.8% |
| 3 | fantasy_pts_rolling_5g_avg | time_series | 1.6% |
| 4 | fantasy_pts_career_max | time_series | 1.1% |
| 5 | fantasy_pts_season_max | time_series | 0.6% |
| 6 | fantasy_pts_lag_1 | time_series | 0.5% |
| 7 | home_game | contextual | 0.5% |
| 8 | tov_rolling_5g_avg | time_series | 0.5% |
| 9 | tov_rolling_20g_avg | time_series | 0.5% |
| 10 | draft_pick_numeric | player_specific | 0.5% |
| 11 | reb_career_max | time_series | 0.5% |
| 12 | tov_season_max | time_series | 0.4% |
| 13 | opponent_win_pct | contextual | 0.4% |
| 14 | days_rest | contextual | 0.4% |
| 15 | stl_career_max | time_series | 0.4% |
| 16 | pts_career_max | time_series | 0.4% |
| 17 | fantasy_pts_season_cumulative_total | time_series | 0.4% |
| 18 | pts_season_max | time_series | 0.4% |
| 19 | tov_lag_1 | time_series | 0.4% |
| 20 | pts_season_cumulative_total | time_series | 0.3% |

## Feature Family Share By Target

| Target | Time-Series | Contextual | Player-Specific | Other |
| --- | --- | --- | --- | --- |
| 3PM | 89.3% | 2.7% | 8.0% | 0.0% |
| AST | 93.5% | 2.1% | 4.5% | 0.0% |
| BLK | 87.4% | 3.9% | 8.7% | 0.0% |
| FANTASY_PTS | 95.2% | 1.9% | 3.0% | 0.0% |
| PTS | 93.1% | 1.9% | 5.0% | 0.0% |
| REB | 89.8% | 2.0% | 8.2% | 0.0% |
| STL | 83.5% | 3.3% | 13.2% | 0.0% |
| TOV | 89.9% | 2.4% | 7.6% | 0.0% |

## Key Insights

Time-series features are the largest feature family for 8 of the 8 targets. This supports keeping the PySpark feature engineering stage because the rolling, lag, season, and career-history features are doing much of the predictive work.

The strongest time-series reliance is for `FANTASY_PTS` at 95.2% of gain importance. The weakest is `STL` at 83.5%, which suggests that some targets depend more on context, player profile, or noisy game-level variation.

Fantasy points importance is useful as the main portfolio signal because it combines several box-score outcomes. If its top features are mostly recent-form or history features, that validates the project design: the model is learning from a player's recent and longer-term performance rather than from same-game leakage columns.

Importance scores should be treated as model diagnostics, not causal explanations. Correlated rolling features can share importance unevenly, so the ranking is best read as a directional view of what the model found useful.
