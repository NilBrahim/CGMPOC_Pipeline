# scripts/09_pgr_tree.py
# Interpretable binary PGR classifier (Large vs Small Δpeak) using a shallow Decision Tree.
# Uses AUTO threshold: Large if delta_peak >= dataset median.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, roc_auc_score
)

from src.config import DATA_DERIVED, REPORTS, ID_COL

IN_CSV     = DATA_DERIVED / "meal_outcomes.csv"
OUT_PRED   = DATA_DERIVED / "pgr_tree_predictions.csv"
OUT_FOLD   = DATA_DERIVED / "pgr_tree_folds.csv"
OUT_RULES  = DATA_DERIVED / "pgr_tree_rules.txt"
OUT_CM_PNG = DATA_DERIVED / "pgr_tree_confusion.png"
OUT_HTML   = REPORTS / "PGR_Tree_summary.html"
OUT_TREE   = REPORTS / "pgr_tree.png"

TARGET      = "delta_peak"
LABEL_COL   = "label_large"   # 1=Large, 0=Small
THRESH_MODE = "auto"          # <-- AUTO median Δpeak
THRESH_VAL  = 60.0            # ignored when THRESH_MODE="auto"

FEATS = [
    "carbs_g","protein_g","fat_g","fiber_g","calories",
    "baseline_mgdl","baseline_sd",
    "gap_prev_meal_hr","hour",
    "steps_prev_120m","hr_mean_prev_60m",
]

def label_large_small(df: pd.DataFrame) -> pd.Series:
    if THRESH_MODE == "auto":
        thr = float(df[TARGET].median())
    else:
        thr = float(THRESH_VAL)
    print(f"Labeling Large vs Small using threshold Δpeak = {thr:.1f} mg/dL "
          f"({'AUTO median' if THRESH_MODE=='auto' else 'FIXED'})")
    return (df[TARGET] >= thr).astype(int)

def impute_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in ["steps_prev_120m","hr_mean_prev_60m"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    for c in X.columns:
        if c not in ["steps_prev_120m","hr_mean_prev_60m"]:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
    return X

def main():
    print("Reading:", IN_CSV)
    df = pd.read_csv(IN_CSV, parse_dates=["meal_time"])

    keep = [ID_COL, "meal_time", TARGET] + [c for c in FEATS if c in df.columns]
    df = df[keep].dropna(subset=[TARGET]).copy()

    X = df[[c for c in FEATS if c in df.columns]].copy()
    X = impute_features(X)

    df[LABEL_COL] = label_large_small(df)
    y    = df[LABEL_COL].values.astype(int)
    pids = df[ID_COL].values

    pos_rate = y.mean()
    print(f"N meals: {len(df)}, Pos rate (Large): {pos_rate:.2f}")

    tree = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=10,
        class_weight="balanced", random_state=0
    )

    unique_pids = np.unique(pids)
    fold_rows, all_preds = [], []

    for pid in unique_pids:
        test_idx, train_idx = (pids == pid), (pids != pid)
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test,  y_test  = X.iloc[test_idx],  y[test_idx]

        if len(np.unique(y_train)) < 2 or len(X_test) == 0:
            print(f"[warn] Skipping pid={pid}: insufficient class diversity or test size.")
            continue

        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)

        try:
            y_prob = tree.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            y_prob = np.full_like(y_test, np.nan, dtype=float)
            auc = np.nan

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="binary", zero_division=0)
        bacc = balanced_accuracy_score(y_test, y_pred)

        fold_rows.append({
            "left_out_pid": int(pid),
            "n_test_meals": int(test_idx.sum()),
            "accuracy": float(acc),
            "f1": float(f1),
            "balanced_acc": float(bacc),
            "roc_auc": float(auc) if np.isfinite(auc) else np.nan,
        })

        all_preds.append(pd.DataFrame({
            "Person_ID": df.loc[test_idx, ID_COL].astype(int).values,
            "meal_time": df.loc[test_idx, "meal_time"].dt.strftime("%Y-%m-%d %H:%M").values,
            "y_true": y_test, "y_pred": y_pred, "y_prob": y_prob,
        }))

    folds = pd.DataFrame(fold_rows).sort_values("left_out_pid")
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    if not folds.empty:
        overall = {
            "acc_mean":  folds["accuracy"].mean(),
            "f1_mean":   folds["f1"].mean(),
            "bacc_mean": folds["balanced_acc"].mean(),
            "auc_mean":  folds["roc_auc"].mean(skipna=True),
        }
    else:
        overall = {"acc_mean": np.nan, "f1_mean": np.nan, "bacc_mean": np.nan, "auc_mean": np.nan}

    folds.to_csv(OUT_FOLD, index=False)
    preds.to_csv(OUT_PRED, index=False)
    print(f"Saved per-fold metrics -> {OUT_FOLD}")
    print(f"Saved predictions      -> {OUT_PRED}")

    # Train one final tree on ALL data (for rules/diagram)
    if len(np.unique(y)) >= 2:
        tree.fit(X, y)
        Path(OUT_RULES).write_text(export_text(tree, feature_names=list(X.columns), decimals=3), encoding="utf-8")
        print(f"Saved rules -> {OUT_RULES}")

        plt.figure(figsize=(10, 6))
        plot_tree(
            tree,
            feature_names=list(X.columns),
            class_names=["Small","Large"],
            filled=True, rounded=True, impurity=False, proportion=True
        )
        plt.tight_layout()
        plt.savefig(OUT_TREE, dpi=160)
        plt.close()
        print(f"Saved tree diagram -> {OUT_TREE}")

    # Confusion matrix from LOPO predictions
    if not preds.empty and {"y_true","y_pred"}.issubset(preds.columns):
        cm = confusion_matrix(preds["y_true"], preds["y_pred"], labels=[0,1])
        cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        plt.figure(figsize=(4.2,3.8))
        plt.imshow(cmn, aspect="auto")
        plt.title("Confusion Matrix (rows=true, cols=pred)")
        plt.xticks([0,1], ["Small","Large"])
        plt.yticks([0,1], ["Small","Large"])
        for i in range(2):
            for j in range(2):
                val = cmn[i, j]
                txt = "—" if not np.isfinite(val) else f"{val*100:.1f}%"
                plt.text(j, i, txt, ha="center", va="center")
        plt.tight_layout()
        plt.savefig(OUT_CM_PNG, dpi=160)
        plt.close()
        print(f"Saved confusion matrix -> {OUT_CM_PNG}")

    # HTML summary
    folds_html = folds.to_html(index=False, border=0) if not folds.empty else "<p>No valid folds.</p>"
    preds_html = preds.head(30).to_html(index=False, border=0) if not preds.empty else "<p>No predictions to show.</p>"
    rules_html = "<pre style='white-space:pre-wrap'>" + (Path(OUT_RULES).read_text(encoding="utf-8") if Path(OUT_RULES).exists() else "No rules.") + "</pre>"

    html = f"""
<!doctype html>
<html>
<head><meta charset="utf-8"><title>PGR POC — Shallow Decision Tree (Δpeak)</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
h1, h2 {{ margin: 8px 0; }}
small {{ color:#666; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; }}
img {{ max-width: 680px; height: auto; border: 1px solid #eee; border-radius: 8px; padding: 4px; }}
</style>
</head>
<body>
  <h1>PGR POC — Decision Tree (Large vs Small Δpeak)</h1>
  <small>Validation: LOPO by participant; Model: depth=3 DecisionTree (class_weight=balanced)</small>

  <h2>Per-fold metrics</h2>
  {folds_html}
  <p><b>Overall</b> — Accuracy: {overall['acc_mean'] if np.isfinite(overall['acc_mean']) else 'N/A':.3f}
     &nbsp; F1: {overall['f1_mean'] if np.isfinite(overall['f1_mean']) else 'N/A':.3f}
     &nbsp; Balanced Acc: {overall['bacc_mean'] if np.isfinite(overall['bacc_mean']) else 'N/A':.3f}
     &nbsp; ROC-AUC: {overall['auc_mean'] if np.isfinite(overall['auc_mean']) else 'N/A':.3f}</p>

  <h2>Tree diagram</h2>
  <img src="pgr_tree.png" alt="pgr tree">

  <h2>Confusion Matrix (LOPO predictions)</h2>
  <img src="../data/derived/{OUT_CM_PNG.name}" alt="confusion matrix">

  <h2>Rules (text)</h2>
  {rules_html}

  <h2>Sample predictions (first 30)</h2>
  {preds_html}
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Saved HTML -> {OUT_HTML}")

if __name__ == "__main__":
    main()