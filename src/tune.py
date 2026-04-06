import logging
import os
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC       = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

N_TRIALS   = 30
CV_FOLDS   = 3
RANDOM_STATE = 42


def load_data():
    X_train = pd.read_csv(os.path.join(PROC, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(PROC, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROC, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(PROC, "y_test.csv")).squeeze()
    return X_train, X_test, y_train, y_test


def baseline_auc(X_train, y_train, X_test, y_test):
    """AUC of the default XGBoost from model.py (scale_pos_weight only)."""
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = xgb.XGBClassifier(
        scale_pos_weight=neg / pos,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


def make_objective(X_train, y_train):
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":  neg / pos,
            "random_state":      RANDOM_STATE,
            "eval_metric":       "logloss",
            "verbosity":         0,
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    return objective


def plot_history(study):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    trials = study.trials
    trial_nums = [t.number + 1 for t in trials if t.value is not None]
    values     = [t.value      for t in trials if t.value is not None]
    best_so_far = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(trial_nums, values, s=30, alpha=0.55,
               color=sns.color_palette("Blues")[3], label="Trial AUC-ROC")
    ax.plot(trial_nums, best_so_far, linewidth=2,
            color=sns.color_palette("Blues_d")[4], label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV AUC-ROC")
    ax.set_title("Optuna Optimization History — XGBoost")
    ax.legend()
    path = os.path.join(FIGURES_DIR, "tuning_history.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def main():
    X_train, X_test, y_train, y_test = load_data()

    print("Computing baseline AUC-ROC on test set...")
    auc_before = baseline_auc(X_train, y_train, X_test, y_test)

    print(f"Running Optuna ({N_TRIALS} trials, {CV_FOLDS}-fold CV)...")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(make_objective(X_train, y_train), n_trials=N_TRIALS,
                   show_progress_bar=False)

    best_params = study.best_params
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    tuned_model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=neg / pos,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )
    tuned_model.fit(X_train, y_train)
    auc_after = roc_auc_score(y_test, tuned_model.predict_proba(X_test)[:, 1])

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_model_tuned.pkl")
    joblib.dump(tuned_model, model_path)

    chart_path = plot_history(study)

    # ── Output ─────────────────────────────────────────────────────────────
    print("\n── Best Parameters ──────────────────────────────────────")
    for k, v in best_params.items():
        print(f"  {k:<22} {v}")

    print("\n── AUC-ROC ──────────────────────────────────────────────")
    delta = auc_after - auc_before
    print(f"  Before tuning : {auc_before:.4f}")
    print(f"  After tuning  : {auc_after:.4f}  ({'+' if delta >= 0 else ''}{delta:.4f})")

    print(f"\nSaved model   : {model_path}")
    print(f"Saved chart   : {chart_path}")


if __name__ == "__main__":
    main()
