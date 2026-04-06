# US Flight Delay Prediction

A machine learning project that predicts whether a US domestic flight will be delayed by 15 or more minutes, using the 2015 DOT On-Time Performance dataset.

## Overview

This project covers end-to-end ML pipeline development including data loading, feature engineering, model training, hyperparameter optimization, and an interactive dashboard for exploring predictions.

## Dataset

- Source: [2015 Flight Delays and Cancellations — Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays)
- Files: `flights.csv`, `airlines.csv`, `airports.csv`
- Place raw data files in `data/raw/` before running the pipeline.

## Project Structure

```
flight-delay-prediction/
├── data/
│   ├── raw/          # Original downloaded files (not versioned)
│   └── processed/    # Cleaned and feature-engineered datasets
├── figures/          # Saved plots and visualizations
├── notebooks/        # Exploratory analysis notebooks
├── src/              # Source code modules
├── tests/            # Unit tests
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python src/data_loader.py
```
