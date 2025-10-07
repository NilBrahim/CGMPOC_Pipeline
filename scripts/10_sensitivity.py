# scripts/10_sensitivity.py
# Sensitivity of LOPO performance to (a) Δpeak threshold and (b) window length.
# We vary: THRESH in {40,50,60,70} mg/dL and WINDOW in {90,120,150} min.
# For WINDOW != 120, we approximate by re-labeling with iauc_0_120 vs. delta_peak rules:
#   - For simplicity in this POC, we ONLY vary the threshold on delta_peak (no re-windowing computation),
#     and produce a separate sweep over thresholds.
#   - Optionally, you can add iauc thresholding or re-windowing later by regenerating meal_outcomes.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from src.config import DATA_DERIVED, REPORTS, ID_COL

IN_CSV = DATA_DERIVED / "meal_outcomes.csv"
OUT_CSV = DATA_DERIVED / "pgr_sensitivity_thresholds.csv"
OUT_PNG = REPORTS / "pgr_sensitivity_thresholds.png"

# We’ll sweep only the Δpeak threshold here (simple, no recompute of outcomes)
THRESHOLDS = [40.0, 50.0, 60.0, 70.0]

FEATS = [
    "carbs_g","protein_g","fat_g","fiber_g","calories",
    "baseline_mgdl","baseline_sd",
    "gap_prev_meal_hr","hour",
    "steps_prev_120m","hr_mean_prev_60m",
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[[c for c in FEATS if c in df.columns]].copy()
    # activity → 0, others → median
    for c in ["steps_prev_120m","hr_mean_prev_60m"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    for c in X.columns:
        if c not in ["steps_prev_120m","hr_mean_prev_60m"]:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
    return X

def eval_threshold(df: pd.DataFrame, thr: float) -> dict:
    y = (df["delta_peak"] >= thr).astype(int).values
    pids = df[ID_COL].values
    X = build_features(df)

    # if only one class -> skip
    if len(np.unique(y)) < 2:
        return {"threshold": thr, "acc": np.nan, "f1": np.nan, "bacc": np.nan, "auc": np.nan, "pos_rate": y.mean()}

    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, class_weight="balanced", random_state=0)

    accs, f1s, baccs, aucs = [], [], [], []
    unique_pids = np.unique(pids)
    for pid in unique_pids:
        test_idx = (pids == pid)
        train_idx = ~test_idx

        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test,  y_test  = X.iloc[test_idx],  y[test_idx]

        if len(np.unique(y_train)) < 2 or len(X_test) == 0:
            continue

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # probs for AUC if available
        try:
            y_prob = clf.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = np.nan

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="binary", zero_division=0))
        baccs.append(balanced_accuracy_score(y_test, y_pred))
        aucs.append(auc)

    return {
        "threshold": thr,
        "acc": np.nanmean(accs) if accs else np.nan,
        "f1":  np.nanmean(f1s)  if f1s  else np.nan,
        "bacc":np.nanmean(baccs)if baccs else np.nan,
        "auc": np.nanmean(aucs) if aucs else np.nan,
        "pos_rate": y.mean(),
    }

def main():
    print("Reading:", IN_CSV)
    df = pd.read_csv(IN_CSV, parse_dates=["meal_time"])
    df = df.dropna(subset=["delta_peak"]).copy()

    rows = []
    for thr in THRESHOLDS:
        res = eval_threshold(df, thr)
        print(f"thr={thr:.0f} → acc={res['acc']:.3f}, f1={res['f1']:.3f}, bacc={res['bacc']:.3f}, auc={res['auc']:.3f}, pos_rate={res['pos_rate']:.2f}")
        rows.append(res)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)

    # simple plot: metrics vs threshold
    plt.figure(figsize=(7,4.5))
    for key, label in [("acc","Accuracy"), ("f1","F1"), ("bacc","Balanced Acc"), ("auc","ROC-AUC")]:
        plt.plot(out["threshold"], out[key], marker="o", label=label)
    plt.xlabel("Δpeak threshold (mg/dL)")
    plt.ylabel("Score")
    plt.title("LOPO performance vs Δpeak threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    plt.close()
    print("Saved:", OUT_PNG)

if __name__ == "__main__":
    main()