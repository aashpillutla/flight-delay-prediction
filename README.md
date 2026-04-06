# US Flight Delay Prediction

A machine learning project that predicts whether a US domestic flight will be delayed by 15 or more minutes, using the 2015 DOT On-Time Performance dataset. Includes a full ML pipeline (data loading → feature engineering → modelling → evaluation), a prescriptive booking optimizer, and an interactive Streamlit dashboard.

## Dataset

- **Source:** [2015 Flight Delays and Cancellations — Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays)
- **Files:** `flights.csv`, `airlines.csv`, `airports.csv`
- **Sample size:** 250,000 flights (stratified 80/20 train/test split)
- **Target:** `delayed` — 1 if `ARRIVAL_DELAY ≥ 15 min`, else 0 (~18.6% positive rate)

## Running
git clone https://github.com/aashpillutla/flight-delay-prediction.git
cd flight-delay-prediction
pip install -r requirements.txt
streamlit run src/dashboard.py


## Dashboard

- **Section 1 — Flight Reliability Explorer:** delay rate by carrier, hour, month, day, top airports, and a US scatter-geo map coloured by delay rate. All charts respond to sidebar filters.
- **Section 2 — Delay Predictor:** enter a specific flight and get a predicted delay probability with a colour-coded badge and top 3 risk factors.
- **Section 3 — Smart Booking Optimizer:**
  - *Optimal Flight* — finds the lowest-risk (carrier, hour) combination within a preferred window and risk tolerance, with a sensitivity chart.
  - *Best Airlines* — ranks all carriers on a route by predicted delay % subject to a minimum flight-frequency constraint.
  - *Best Airport* — compares origin airports in the same city (e.g. JFK / LGA / EWR) for a given route.
