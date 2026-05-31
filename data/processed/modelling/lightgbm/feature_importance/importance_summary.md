# Feature Importance Summary

This summary uses LightGBM gain importance from the saved default models. Gain measures how much a feature improves tree splits when it is used, so larger values mean the model relied on that feature more.

## Top 20 Fantasy Points Features

| Rank | Feature | Feature Family | Gain Share |
| 1 | fantasy_pts_rolling_20g_avg | time_series | 65.3% |
| 2 | fantasy_pts_rolling_10g_avg | time_series | 21.5% |
| 3 | fantasy_pts_rolling_5g_avg | time_series | 4.1% |
| 4 | fantasy_pts_career_max | time_series | 0.7% |
| 5 | fantasy_pts_lag_1 | time_series | 0.7% |
| 6 | opponent_win_pct | contextual | 0.7% |
| 7 | fantasy_pts_season_max | time_series | 0.6% |
| 8 | days_rest | contextual | 0.3% |
| 9 | tov_rolling_5g_avg | time_series | 0.3% |
| 10 | tov_rolling_20g_avg | time_series | 0.2% |
| 11 | draft_pick_numeric | player_specific | 0.2% |
| 12 | opponent_def_rating | contextual | 0.2% |
| 13 | tov_rolling_10g_avg | time_series | 0.2% |
| 14 | pts_lag_1 | time_series | 0.2% |
| 15 | tov_lag_1 | time_series | 0.1% |
| 16 | fantasy_pts_season_min | time_series | 0.1% |
| 17 | pts_season_max | time_series | 0.1% |
| 18 | home_game | contextual | 0.1% |
| 19 | fantasy_pts_rolling_10g_std | time_series | 0.1% |
| 20 | tov_season_cumulative_total | time_series | 0.1% |

## Feature Family Share By Target

| Target | Time-Series | Contextual | Player-Specific | Other |
| 3PM | 97.3% | 1.4% | 1.3% | 0.0% |
| AST | 98.4% | 1.0% | 0.6% | 0.0% |
| BLK | 92.7% | 3.7% | 3.6% | 0.0% |
| FANTASY_PTS | 98.0% | 1.4% | 0.6% | 0.0% |
| PTS | 98.3% | 1.0% | 0.7% | 0.0% |
| REB | 97.7% | 0.9% | 1.4% | 0.0% |
| STL | 91.8% | 3.2% | 4.9% | 0.0% |
| TOV | 96.8% | 1.6% | 1.7% | 0.0% |

## Key Insights

Time-series features are the largest feature family for 8 of the 8 targets, similar to the analysis done in XGBoost.

Compared with XGBoost, LightGBM places an even larger share of gain importance on time-series features across every target.

The strongest time-series reliance is for `AST` at 98.4% of gain importance. The weakest is `STL` at 91.8%, which suggests that some targets depend more on context, player profile, or noisy game-level variation.

The ranking best reflects what the model found most useful as there could be other features that contributed to the predictions.
