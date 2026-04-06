import logging
import os
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["PYTHONWARNINGS"] = "ignore"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
)

import xgboost as xgb
import lightgbm as lgb

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_data():
    X_train = pd.read_csv(os.path.join(PROC, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(PROC, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROC, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(PROC, "y_test.csv")).squeeze()
    return X_train, X_test, y_train, y_test


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )
    return {
        "Model":     name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1":        f1_score(y_test, y_pred, zero_division=0),
        "AUC-ROC":   roc_auc_score(y_test, y_prob),
        "_prob":     y_prob,
    }


def build_models(neg, pos):
    scale = neg / pos
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            scale_pos_weight=scale, random_state=42,
            eval_metric="logloss", verbosity=0, use_label_encoder=False,
        ),
        "LightGBM": lgb.LGBMClassifier(
            is_unbalance=True, random_state=42, verbose=-1,
        ),
    }


def plot_roc_curves(results, y_test):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("tab10", len(results))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.50)")

    for (res, color) in zip(results, palette):
        fpr, tpr, _ = roc_curve(y_test, res["_prob"])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{res['Model']} (AUC = {res['AUC-ROC']:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    path = os.path.join(FIGURES_DIR, "roc_curves.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def print_table(results):
    rows = []
    for r in results:
        rows.append({k: v for k, v in r.items() if k != "_prob"})
    df = pd.DataFrame(rows).set_index("Model")
    float_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    fmt = df[float_cols].map(lambda x: f"{x:.4f}")

    col_w = 11
    header = f"{'Model':<22}" + "".join(f"{c:>{col_w}}" for c in float_cols)
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for model_name, row in fmt.iterrows():
        print(f"{model_name:<22}" + "".join(f"{v:>{col_w}}" for v in row))
    print(sep)


def main():
    X_train, X_test, y_train, y_test = load_data()

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    models = build_models(neg, pos)
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))

    print_table(results)

    plot_roc_curves(results, y_test)

    best = max(results, key=lambda r: r["AUC-ROC"])
    best_name = best["Model"]
    best_model = models[best_name]

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)

    print(f"\nBest model : {best_name}  (AUC-ROC = {best['AUC-ROC']:.4f})")
    print(f"Saved      : {model_path}")
    print(f"ROC chart  : {os.path.join(FIGURES_DIR, 'roc_curves.png')}")


if __name__ == "__main__":
    main()
