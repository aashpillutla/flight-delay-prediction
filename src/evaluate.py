import logging
import os
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.join(os.path.dirname(__file__), "..")
PROC        = os.path.join(ROOT, "data", "processed")
FIGURES_DIR = os.path.join(ROOT, "figures")
MODELS_DIR  = os.path.join(ROOT, "models")

SHAP_SAMPLE   = 5000
RANDOM_STATE  = 42
DELAY_COL     = "delayed"

sns.set_theme(style="whitegrid")


# ── Helpers ───────────────────────────────────────────────────────────────────
def save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def load_artifacts():
    model   = joblib.load(os.path.join(MODELS_DIR, "best_model_tuned.pkl"))
    X_test  = pd.read_csv(os.path.join(PROC, "X_test.csv"))
    y_test  = pd.read_csv(os.path.join(PROC, "y_test.csv")).squeeze()
    # Raw test rows (same 80/20 split, random_state=42) for error analysis
    clean   = pd.read_csv(os.path.join(PROC, "flights_clean.csv"), low_memory=False)
    _, raw_test = train_test_split(clean, test_size=0.20,
                                   stratify=clean[DELAY_COL], random_state=RANDOM_STATE)
    raw_test = raw_test.reset_index(drop=True)
    return model, X_test, y_test, raw_test


# ── 1. Confusion matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                xticklabels=["On-time", "Delayed"],
                yticklabels=["On-time", "Delayed"],
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    path = save(fig, "confusion_matrix.png")
    tn, fp, fn, tp = cm.ravel()
    print(f"[confusion_matrix] TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,} → saved {path}")


# ── 2. Precision-recall curve ─────────────────────────────────────────────────
def plot_pr_curve(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precision + recall) == 0, 0,
        2 * precision * recall / (precision + recall)
    )
    best_idx = np.argmax(f1_scores[:-1])   # last point has no threshold
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recall, precision, linewidth=2,
            color=sns.color_palette("Blues_d")[4], label="PR curve")
    ax.scatter(recall[best_idx], precision[best_idx],
               s=120, zorder=5, color="tomato",
               label=f"Optimal threshold = {best_thresh:.2f}  (F1 = {best_f1:.3f})")
    baseline = y_true.mean()
    ax.axhline(baseline, linestyle="--", linewidth=1, color="grey",
               label=f"No-skill baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    path = save(fig, "pr_curve.png")
    print(f"[pr_curve] Optimal threshold = {best_thresh:.2f}  "
          f"(Precision={precision[best_idx]:.3f}, Recall={recall[best_idx]:.3f}, F1={best_f1:.3f})"
          f" → saved {path}")
    return best_thresh


# ── 3 & 4. SHAP ───────────────────────────────────────────────────────────────
def plot_shap(model, X_test):
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_test), size=min(SHAP_SAMPLE, len(X_test)), replace=False)
    X_sample = X_test.iloc[idx].reset_index(drop=True)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot (top 12)
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X_sample, max_display=12,
                      show=False, plot_size=None)
    plt.title("SHAP Summary — Top 12 Features", pad=12)
    plt.tight_layout()
    path = save(fig, "shap_summary.png")
    print(f"[shap_summary] saved {path}")

    # Top 2 features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top2_idx = np.argsort(mean_abs)[::-1][:2]
    top2_features = X_sample.columns[top2_idx].tolist()

    for rank, feat in enumerate(top2_features, start=1):
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(feat, shap_values, X_sample,
                             ax=ax, show=False)
        ax.set_title(f"SHAP Dependence — {feat}")
        plt.tight_layout()
        path = save(fig, f"shap_dep_{rank}.png")
        print(f"[shap_dep_{rank}] feature='{feat}' → saved {path}")

    return top2_features


# ── 5. Error analysis ─────────────────────────────────────────────────────────
def error_analysis(y_true, y_pred, raw_test):
    airport_col = "ORIGIN_AIRPORT_NAME" if "ORIGIN_AIRPORT_NAME" in raw_test.columns else "ORIGIN_AIRPORT"
    df = raw_test[["AIRLINE_NAME", airport_col]].copy()
    df["y_true"] = y_true.values
    df["y_pred"] = y_pred

    def worst_groups(group_col, label, n=5):
        rows = []
        for name, grp in df.groupby(group_col):
            if len(grp) < 30:          # skip thin groups
                continue
            if grp["y_true"].sum() < 5:  # skip groups with almost no positives
                continue
            f1 = f1_score(grp["y_true"], grp["y_pred"], zero_division=0)
            rows.append({"name": name, "n": len(grp), "f1": f1})
        worst = sorted(rows, key=lambda r: r["f1"])[:n]
        col_w_name = max(len(r["name"]) for r in worst) + 2
        header = f"  {'Name':<{col_w_name}} {'Flights':>8}  {'F1':>6}"
        sep    = "  " + "-" * (len(header) - 2)
        print(f"\n  Worst 5 {label} by F1")
        print(sep)
        print(header)
        print(sep)
        for r in worst:
            print(f"  {r['name']:<{col_w_name}} {r['n']:>8,}  {r['f1']:>6.3f}")

    worst_groups("AIRLINE_NAME", "carriers")
    worst_groups(airport_col, "origin airports")


# ── 6. Sensitivity: threshold 15 vs 30 ───────────────────────────────────────
def sensitivity_analysis(model, X_test, raw_test):
    clean = pd.read_csv(os.path.join(PROC, "..", "..", "data", "processed",
                                     "flights_clean.csv"), low_memory=False)
    _, raw_test_full = train_test_split(
        clean, test_size=0.20, stratify=clean[DELAY_COL], random_state=RANDOM_STATE
    )
    raw_test_full = raw_test_full.reset_index(drop=True)

    y_prob = model.predict_proba(X_test)[:, 1]

    results = []
    for thresh_min in [15, 30]:
        y_true = (raw_test_full["ARRIVAL_DELAY"] >= thresh_min).astype(int)
        # Re-derive predictions using default 0.5 decision boundary
        y_pred = (y_prob >= 0.5).astype(int)
        results.append({
            "Delay threshold": f">= {thresh_min} min",
            "Positives":       f"{y_true.sum():,} ({y_true.mean()*100:.1f}%)",
            "Accuracy":        f"{accuracy_score(y_true, y_pred):.4f}",
            "Precision":       f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
            "Recall":          f"{recall_score(y_true, y_pred, zero_division=0):.4f}",
            "F1":              f"{f1_score(y_true, y_pred, zero_division=0):.4f}",
            "AUC-ROC":         f"{roc_auc_score(y_true, y_prob):.4f}",
        })

    cols = list(results[0].keys())
    col_w = [max(len(c), max(len(r[c]) for r in results)) + 2 for c in cols]
    header = "".join(f"{c:<{w}}" for c, w in zip(cols, col_w))
    sep    = "-" * len(header)
    print(f"\n  Sensitivity: Delay Threshold Comparison")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")
    for r in results:
        print("  " + "".join(f"{r[c]:<{w}}" for c, w in zip(cols, col_w)))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading model and data...")
    model, X_test, y_test, raw_test = load_artifacts()

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    print(f"Tuned model AUC-ROC on test set: {auc:.4f}\n")

    print("1/6  Confusion matrix...")
    plot_confusion_matrix(y_test, y_pred_default)

    print("\n2/6  Precision-recall curve...")
    best_thresh = plot_pr_curve(y_test, y_prob)

    print("\n3/6  SHAP summary + dependence plots...")
    top2 = plot_shap(model, X_test)

    print("\n4/6  Error analysis...")
    error_analysis(y_test, y_pred_default, raw_test)

    print("\n5/6  Sensitivity analysis...")
    sensitivity_analysis(model, X_test, raw_test)

    print("\nDone. All figures saved to figures/")


if __name__ == "__main__":
    main()
