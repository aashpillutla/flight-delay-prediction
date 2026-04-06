# US Flight Delay Prediction

A machine learning project that predicts whether a US domestic flight will be delayed by 15 or more minutes, using the 2015 DOT On-Time Performance dataset. Includes a full ML pipeline (data loading → feature engineering → modelling → evaluation), a prescriptive booking optimizer, and an interactive Streamlit dashboard.

## Dataset

- **Source:** [2015 Flight Delays and Cancellations — Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays)
- **Files:** `flights.csv`, `airlines.csv`, `airports.csv`
- **Sample size:** 250,000 flights (stratified 80/20 train/test split)
- **Target:** `delayed` — 1 if `ARRIVAL_DELAY ≥ 15 min`, else 0 (~18.6% positive rate)

Place the three CSV files in `data/raw/` before running the pipeline.

## Project Structure

```
flight-delay-prediction/
├── data/
│   ├── raw/                  # Source CSVs (not versioned)
│   └── processed/            # Generated train/test splits (not versioned)
├── figures/                  # All saved charts (EDA, model evaluation, tuning)
├── models/                   # Saved model artifacts
│   ├── best_model.pkl
│   └── best_model_tuned.pkl
├── src/
│   ├── data_loader.py        # Sampling, merging, cleaning, target creation
│   ├── eda.py                # 7 exploratory visualisations
│   ├── feature_eng.py        # Feature engineering + train/test export
│   ├── model.py              # Train & compare 4 models
│   ├── tune.py               # Optuna hyperparameter tuning (XGBoost)
│   ├── evaluate.py           # Confusion matrix, PR curve, SHAP, error analysis
│   ├── optimizer.py          # Prescriptive booking optimizer
│   └── dashboard.py          # Streamlit app
└── .gitignore
```

## Pipeline

Run steps in order:

```bash
python3 src/data_loader.py    # → data/processed/flights_clean.csv
python3 src/eda.py            # → figures/  (7 charts)
python3 src/feature_eng.py    # → data/processed/X_train, X_test, y_train, y_test
python3 src/model.py          # → models/best_model.pkl  (comparison table)
python3 src/tune.py           # → models/best_model_tuned.pkl  (Optuna, 30 trials)
python3 src/evaluate.py       # → figures/  (confusion matrix, PR curve, SHAP)
streamlit run src/dashboard.py
```

## Features (10)

| Feature | Description |
|---|---|
| `hour` | Departure hour (0–23) extracted from `DEPARTURE_TIME` |
| `time_of_day` | Ordinal: morning / afternoon / evening / night |
| `day_of_week` | 1 (Mon) – 7 (Sun) |
| `is_weekend` | 1 if Saturday or Sunday |
| `month` | 1–12 |
| `is_holiday` | 1 if date is a US federal holiday or eve/day-after |
| `distance_bin` | short (<500 mi) / medium / long (>1500 mi) |
| `carrier_delay_rate` | Historical delay rate for the airline (train set only) |
| `origin_delay_rate` | Historical delay rate for the origin airport (train set only) |
| `route_delay_rate` | Historical delay rate for the origin→destination pair (train set only) |

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.630 | 0.269 | 0.575 | 0.367 | 0.646 |
| Random Forest | 0.806 | 0.434 | 0.130 | 0.200 | 0.674 |
| XGBoost | 0.637 | 0.285 | 0.630 | 0.392 | **0.681** |
| LightGBM | 0.621 | 0.277 | 0.645 | 0.388 | 0.676 |

**XGBoost** selected as best model (highest AUC-ROC). After Optuna tuning (30 trials, 3-fold CV): AUC-ROC improved from 0.6812 → **0.6838**.

Top SHAP features: `route_delay_rate`, `hour`, `carrier_delay_rate`.

## Dashboard

```bash
streamlit run src/dashboard.py
```

Three sections:

- **Section 1 — Flight Reliability Explorer:** delay rate by carrier, hour, month, day of week, top 15 airports, and a scatter geo map of the US coloured by delay rate. All charts respond to sidebar airline/airport filters.
- **Section 2 — Delay Predictor:** enter a specific flight (origin, destination, carrier, time) and get a predicted delay probability, colour-coded reliability badge, and top 3 risk factors.
- **Section 3 — Smart Booking Optimizer:**
  - *Optimal Flight* — finds the lowest-risk (carrier, departure hour) combination within a preferred time window and risk tolerance, with a sensitivity chart showing how many options open up as tolerance is relaxed.
  - *Best Airlines* — ranks all carriers on a route by predicted delay %, subject to a minimum flight-frequency constraint.
  - *Best Airport* — compares origin airports in the same city (e.g. JFK / LGA / EWR) for a given route.

## Dependencies

```bash
pip3 install pandas numpy scikit-learn xgboost lightgbm optuna shap \
             matplotlib seaborn joblib streamlit plotly
```

XGBoost and LightGBM require `libomp` on macOS (`brew install libomp`).
