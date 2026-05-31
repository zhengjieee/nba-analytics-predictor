# Model Evaluation Summary

This summarises the default LightGBM models trained for the eight prediction targets. The baseline used for comparison is a simple train-average baseline: for each target, it predicts the average value from the training set for every test row.

## Metrics

### MAE

| Target | Baseline MAE | LightGBM MAE | MAE Improvement |
| FANTASY_PTS | 11.783 | 8.068 | 3.715 |
| PTS | 7.246 | 4.963 | 2.283 |
| REB | 2.758 | 2.001 | 0.757 |
| AST | 2.110 | 1.462 | 0.648 |
| STL | 0.792 | 0.766 | 0.026 |
| BLK | 0.686 | 0.563 | 0.122 |
| TOV | 1.203 | 0.949 | 0.254 |
| 3PM | 1.252 | 0.996 | 0.256 |

### RMSE

| Target | Baseline RMSE | LightGBM RMSE | RMSE Improvement |
| FANTASY_PTS | 14.491 | 10.228 | 4.263 |
| PTS | 8.892 | 6.398 | 2.494 |
| REB | 3.434 | 2.592 | 0.842 |
| AST | 2.718 | 1.941 | 0.777 |
| STL | 1.024 | 0.979 | 0.045 |
| BLK | 0.850 | 0.758 | 0.092 |
| TOV | 1.469 | 1.234 | 0.235 |
| 3PM | 1.589 | 1.342 | 0.247 |

### R2

| Target | Baseline R2 | LightGBM R2 | R2 Improvement |
| FANTASY_PTS | -0.028 | 0.488 | 0.516 |
| PTS | -0.024 | 0.470 | 0.494 |
| REB | -0.028 | 0.415 | 0.442 |
| AST | -0.003 | 0.488 | 0.492 |
| STL | -0.001 | 0.085 | 0.086 |
| BLK | -0.010 | 0.196 | 0.206 |
| TOV | -0.023 | 0.279 | 0.302 |
| 3PM | -0.005 | 0.283 | 0.288 |

## Target Quality

Using LightGBM R2 as the main quality signal:

- Well predicted: `FANTASY_PTS`, `AST`, `PTS`, `REB`
- Moderately predicted: `3PM`, `TOV`
- Poorly predicted: `BLK`, `STL`

## Key Insights

LightGBM improves over the train-average baseline for all eight targets based on MAE and RMSE. The strongest targets are fantasy points, assists, points, and rebounds, which likely benefit from recent form, usage, and rolling performance features.

Turnovers and made threes show moderate predictive signal. These outcomes are partly stable, but still affected by matchup, role, shooting variance, and game flow.

Steals and blocks remain the weakest targets. This is expected because they are sparse, high-variance defensive events; many player-games have zero or one, and individual game outcomes are noisy.

Overall, LightGBM is most reliable for volume and usage-driven stats, and least reliable for rare defensive counting stats.
