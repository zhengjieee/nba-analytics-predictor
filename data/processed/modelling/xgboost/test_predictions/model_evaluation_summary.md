# Model Evaluation Summary

This summarises the default XGBoost models trained for the eight prediction targets. The baseline used for comparison is a simple train-average baseline: for each target, it predicts the average value from the training set for every test row.

## Metrics

### MAE

| Target | Baseline MAE | XGBoost MAE | MAE Improvement |
| --- | ---: | ---: | ---: |
| FANTASY_PTS | 11.783 | 8.220 | 3.563 |
| PTS | 7.246 | 5.063 | 2.183 |
| REB | 2.758 | 2.046 | 0.712 |
| AST | 2.110 | 1.497 | 0.612 |
| STL | 0.792 | 0.784 | 0.008 |
| BLK | 0.686 | 0.574 | 0.112 |
| TOV | 1.203 | 0.966 | 0.237 |
| 3PM | 1.252 | 1.024 | 0.228 |

### RMSE

| Target | Baseline RMSE | XGBoost RMSE | RMSE Improvement |
| --- | ---: | ---: | ---: |
| FANTASY_PTS | 14.491 | 10.424 | 4.067 |
| PTS | 8.892 | 6.524 | 2.369 |
| REB | 3.434 | 2.649 | 0.785 |
| AST | 2.718 | 1.991 | 0.727 |
| STL | 1.024 | 1.001 | 0.024 |
| BLK | 0.850 | 0.778 | 0.073 |
| TOV | 1.469 | 1.254 | 0.215 |
| 3PM | 1.589 | 1.376 | 0.213 |

### R2

| Target | Baseline R2 | XGBoost R2 | R2 Improvement |
| --- | ---: | ---: | ---: |
| FANTASY_PTS | -0.028 | 0.468 | 0.496 |
| PTS | -0.024 | 0.449 | 0.473 |
| REB | -0.028 | 0.389 | 0.416 |
| AST | -0.003 | 0.461 | 0.465 |
| STL | -0.001 | 0.045 | 0.046 |
| BLK | -0.010 | 0.155 | 0.165 |
| TOV | -0.023 | 0.255 | 0.278 |
| 3PM | -0.005 | 0.246 | 0.251 |

## Target Quality

Using XGBoost R2 as the main quality signal:

- Well predicted: `FANTASY_PTS`, `AST`, `PTS`
- Moderately predicted: `REB`, `TOV`, `3PM`
- Poorly predicted: `BLK`, `STL`

## Key Insights

XGBoost improves over the train-average baseline for all eight targets based on MAE and RMSE. The strongest targets are fantasy points, assists, and points, which likely benefit from recent form, usage, and rolling performance features.

Rebounds, turnovers, and made threes show moderate predictive signal. These outcomes are partly stable, but still affected by matchup, role, shooting variance, and game flow.

Steals and blocks are the weakest targets. This is expected because they are sparse, high-variance defensive events; many player-games have zero or one, and individual game outcomes are noisy.

Overall, the model is most reliable for volume and usage-driven stats, and least reliable for rare defensive counting stats.
