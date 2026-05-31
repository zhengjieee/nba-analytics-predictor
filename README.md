# NBA Analytics Predictor

NBA fantasy basketball prediction pipeline using `nba_api`, PySpark, Pandas, XGBoost, and LightGBM.

The project collects historical NBA player game logs, builds time-series and contextual features, trains regression models for eight player-stat targets, and compares model performance against a simple baseline.

## Prediction Targets

The models predict eight numeric targets:

- Fantasy points (`FANTASY_PTS`)
- Points (`PTS`)
- Rebounds (`REB`)
- Assists (`AST`)
- Steals (`STL`)
- Blocks (`BLK`)
- Turnovers (`TOV`)
- Three-pointers made (`3PM`)

Each target is trained as a separate regression model. `FANTASY_PTS` is a single score that summarises a player’s game performance by giving positive or negative weights to other stats like points, rebounds and turnovers.

## Setup and Reproduction Guide

This project was built with Python 3.12 on macOS. PySpark also needs Java; Java 17 was used for this project.

### 1. Clone the repository

```bash
git clone https://github.com/zhengjieee/nba-analytics-predictor.git
cd nba-analytics-predictor
```

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

```bash
python -m pip install nba_api pandas numpy pyarrow pyspark xgboost lightgbm matplotlib seaborn jupyter
```

### 4. Configure PySpark to use the virtual environment

```bash
export PYSPARK_PYTHON=$PWD/.venv/bin/python
export PYSPARK_DRIVER_PYTHON=$PWD/.venv/bin/python
```

### 5. Reproduce the pipeline

Run the scripts in order:

```bash
python src/data_collection.py --use-existing-selection
python src/pyspark_features.py --partitions 8
python src/pandas_features.py --full
python src/prepare_modelling.py
python src/train_models.py --model xgboost
python src/train_models.py --model lightgbm
python src/feature_importance.py --model xgboost
python src/feature_importance.py --model lightgbm
```

The notebooks can then be opened in Jupyter or VS Code:

```bash
jupyter notebook
```

## Methodology

The project follows a reproducible data-to-model pipeline:

1. **Data collection**: `nba_api` is used to collect selected current NBA players, career game logs, player metadata, team stats, and opponent defensive context.
2. **PySpark feature engineering**: player game logs are converted into time-series features such as rolling averages, rolling standard deviations, season/career min-max values, cumulative totals, and lag features.
3. **Pandas feature completion**: PySpark features are merged with player metadata and contextual features such as home/away, days rest, opponent strength, player age, draft information, position encoding, and shooting efficiency rates.
4. **Time-aware modelling split**: historical games are sorted chronologically, with the most recent 20% used as the test set. Same-game box-score outcome columns are removed from the feature matrix to avoid data leakage.
5. **Model training and evaluation**: XGBoost and LightGBM are trained separately for all eight targets, then compared against a simple train-average baseline using MAE, RMSE, and R2.
6. **Model interpretation and visualisation**: gain-based feature importance and notebook visualisations are used to compare model performance and identify which features drive predictions.

Main scripts:

```text
src/data_collection.py
src/pyspark_features.py
src/pandas_features.py
src/prepare_modelling.py
src/train_models.py
src/feature_importance.py
```

## Results Summary

Both model families beat the train-average baseline across all eight targets. LightGBM performs slightly better than XGBoost across the saved test-set metrics.

For detailed model-specific insights, see the generated Markdown summaries:

- [XGBoost evaluation summary](data/processed/modelling/xgboost/test_predictions/model_evaluation_summary.md)
- [LightGBM evaluation summary](data/processed/modelling/lightgbm/test_predictions/model_evaluation_summary.md)
- [XGBoost feature importance summary](data/processed/modelling/xgboost/feature_importance/importance_summary.md)
- [LightGBM feature importance summary](data/processed/modelling/lightgbm/feature_importance/importance_summary.md)

## Prediction Workflow

Weekly future predictions are intentionally deferred until the next regular-season schedule is available. The 2025-26 regular season has already ended, and the next useful upcoming schedule is 2026-27, which is estimated to be available in mid August.

## File Structure

```text
.
├── .github/workflows/
│   └── ci.yml
├── data/
│   ├── raw/
│   │   ├── selected_players.csv
│   │   ├── player_game_logs.csv
│   │   ├── player_info.csv
│   │   ├── team_stats.csv
│   │   └── opponent_defense.csv
│   └── processed/
│       ├── features/
│       │   ├── features_pyspark.parquet
│       │   ├── features_player_info.parquet
│       │   └── features_full.parquet
│       └── modelling/
│           ├── split/
│           ├── xgboost/
│           │   ├── test_predictions/
│           │   │   ├── model_predictions.parquet
│           │   │   ├── model_metrics.json
│           │   │   └── model_evaluation_summary.md
│           │   └── feature_importance/
│           │       ├── importance_scores.json
│           │       └── importance_summary.md
│           └── lightgbm/
│               ├── test_predictions/
│               │   ├── model_predictions.parquet
│               │   ├── model_metrics.json
│               │   └── model_evaluation_summary.md
│               └── feature_importance/
│                   ├── importance_scores.json
│                   └── importance_summary.md
├── models/
│   ├── xgboost/
│   └── lightgbm/
├── notebooks/
│   ├── eda_analysis.ipynb
│   └── model_visualisations.ipynb
├── src/
│   ├── data_collection.py
│   ├── pyspark_features.py
│   ├── pandas_features.py
│   ├── prepare_modelling.py
│   ├── train_models.py
│   └── feature_importance.py
└── tests/
    ├── test_pyspark_features.py
    ├── test_pandas_features.py
    ├── test_prepare_modelling.py
    ├── test_train_models.py
    └── test_feature_importance.py
```
