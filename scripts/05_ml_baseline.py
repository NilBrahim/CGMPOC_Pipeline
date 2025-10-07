# scripts/05_ml_baseline.py
# Purpose:
# - Classify metabolic state (Healthy / Pre-diabetes / T2D) from filtered daily CGM metrics.
# - Label each row via HbA1c from 'bio' sheet, then run Leave-One-Participant-Out (LOPO) CV.
# - Save: per-fold metrics (acc/macro-F1/bal-acc), all predictions, confusion matrix, and an HTML summary.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, classification_report
)

# project config
from src.config import DATA_RAW, DATA_DERIVED, REPORTS, ID_COL

# ---------- inputs & outputs ----------
DAILY      = DATA_DERIVED / "daily_metrics.csv"          # filtered days (Step 03)
BIO        = DATA_RAW / "CGMacros_dictionary.xlsx"
OUT_PRED   = DATA_DERIVED / "ml_poc_predictions.csv"
OUT_FOLDS  = DATA_DERIVED / "ml_poc_folds.csv"
OUT_CM_PNG = DATA_DERIVED / "ml_poc_confusion_matrix.png"
OUT_HTML   = REPORTS / "ML_POC_summary.html"

BIO_SHEET  = "bio"
HBA1C_COL  = "A1c PDL (Lab)"
BIO_ID_COL = "subject"

print("Reading:", DAILY)
print("Reading:", BIO)

# ---------- load & label ----------
daily = pd.read_csv(DAILY, parse_dates=["date"])
bio   = pd.read_excel(BIO, sheet_name=BIO_SHEET)

# minimal columns from bio
bio_small = bio[[c for c in [BIO_ID_COL, HBA1C_COL] if c in bio.columns]].copy()
bio_small[BIO_ID_COL] = pd.to_numeric(bio_small[BIO_ID_COL], errors="coerce").astype("Int64")

# join HbA1c to daily metrics
daily[ID_COL] = pd.to_numeric(daily[ID_COL], errors="coerce").astype("Int64")
df = daily.merge(bio_small, left_on=ID_COL, right_on=BIO_ID_COL, how="left").drop(columns=[BIO_ID_COL])

# derive label from HbA1c
bins   = [-np.inf, 5.7, 6.4, np.inf]
labels = ["Healthy", "Pre-diabetes", "T2D"]
df["label"] = pd.cut(pd.to_numeric(df[HBA1C_COL], errors="coerce"), bins=bins, labels=labels)

# feature set
feature_cols = ["mean","sd","cv","gmi","tir_pct","below70_pct","above180_pct","below54_pct","above250_pct"]
have = [c for c in feature_cols if c in df.columns]

# drop rows without label or feature values
df = df.dropna(subset=["label"] + have).copy()
if df.empty:
    raise ValueError("No labeled rows available for ML. Check inputs / merged HbA1c!")

print(f"Rows available for ML: {len(df)} across {df[ID_COL].nunique()} participants.")
print(df.groupby(["label", ID_COL]).size().rename("n_days").reset_index().to_string(index=False))

# encode labels
label_to_int = {lbl: i for i, lbl in enumerate(labels)}
int_to_label = {i: lbl for lbl, i in label_to_int.items()}
df["y"] = df["label"].map(label_to_int)

X    = df[have].values.astype(float)
y    = df["y"].values
pids = df[ID_COL].values

# ---------- model pipeline ----------
pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(solver="liblinear", max_iter=1000, class_weight="balanced"))
])

# ---------- Leave-One-Participant-Out CV ----------
fold_rows = []
all_preds = []

unique_pids = np.unique(pids)
for pid in unique_pids:
    test_idx  = (pids == pid)
    train_idx = ~test_idx

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # class presence summary
    train_counts = pd.Series(y_train).value_counts().reindex(range(3), fill_value=0).to_dict()
    test_counts  = pd.Series(y_test).value_counts().reindex(range(3),  fill_value=0).to_dict()

    # if training collapses to <2 classes, skip fold
    if len(np.unique(y_train)) < 2:
        print(f"[warn] Skipping fold pid={pid}: training set has <2 classes. train_counts={train_counts}")
        continue

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    bal_acc  = balanced_accuracy_score(y_test, y_pred)

    fold_rows.append({
        "left_out_pid": int(pid),
        "n_test_days": int(test_idx.sum()),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "balanced_acc": float(bal_acc),
        "train_healthy": int(train_counts.get(0, 0)),
        "train_prediabetes": int(train_counts.get(1, 0)),
        "train_t2d": int(train_counts.get(2, 0)),
        "test_healthy": int(test_counts.get(0, 0)),
        "test_prediabetes": int(test_counts.get(1, 0)),
        "test_t2d": int(test_counts.get(2, 0)),
    })

    # detailed predictions for this fold
    fold_pred = pd.DataFrame({
        "Person_ID": pids[test_idx].astype(int),
        "date":      df.loc[test_idx, "date"].dt.date.values,
        "true_label": [int_to_label[int(v)] for v in y_test],
        "pred_label": [int_to_label[int(v)] for v in y_pred],
    })
    all_preds.append(fold_pred)

# ---------- summarize ----------
folds = pd.DataFrame(fold_rows).sort_values("left_out_pid")
preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

if folds.empty:
    overall_acc = np.nan
    overall_f1  = np.nan
    overall_bal = np.nan
else:
    overall_acc = folds["accuracy"].mean()
    overall_f1  = folds["macro_f1"].mean()
    overall_bal = folds["balanced_acc"].mean()

print("\nPer-fold metrics (LOPO):")
print(folds.to_string(index=False) if not folds.empty else "(no valid folds)")
if np.isfinite(overall_acc):
    print(f"\nOverall (LOPO) — Accuracy: {overall_acc:.3f}, Macro-F1: {overall_f1:.3f}, Balanced Acc: {overall_bal:.3f}")
else:
    print("\nOverall (LOPO): N/A")

# Build confusion matrix from LOPO predictions (preferred). Only fallback if none exist.
use_fallback = preds.empty
if use_fallback:
    print("[info] Using fallback single-fit CM (POC) because LOPO predictions are missing.")
    pipe.fit(X, y)
    y_true = y
    y_pred = pipe.predict(X)
else:
    # Align LOPO predictions with ground truth by merging on (Person_ID, date)
    # 1) take ground truth for those exact rows
    truth = (
        df[[ID_COL, "date", "y"]]
        .assign(date=df["date"].dt.date)
        .rename(columns={"y": "true_y"})
    )
    pred_int = preds.assign(pred_y=preds["pred_label"].map(label_to_int))
    merged = pred_int.merge(truth, on=[ID_COL, "date"], how="inner")

    # keep only rows where we have both y_true and y_pred
    merged = merged.dropna(subset=["true_y", "pred_y"])
    y_true = merged["true_y"].astype(int).values
    y_pred = merged["pred_y"].astype(int).values

cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# ---------- save artifacts ----------
folds.to_csv(OUT_FOLDS, index=False)
preds.to_csv(OUT_PRED, index=False)
print(f"\nSaved per-fold metrics -> {OUT_FOLDS}")
print(f"Saved predictions      -> {OUT_PRED}")

# plot confusion matrix
plt.figure(figsize=(6,5))
plt.imshow(cm_norm, aspect='auto')
plt.title("Confusion Matrix (rows true, cols pred)")
plt.xticks([0,1,2], labels, rotation=0)
plt.yticks([0,1,2], labels)
for i in range(3):
    for j in range(3):
        val = cm_norm[i, j]
        txt = "—" if not np.isfinite(val) else f"{val*100:.1f}%"
        plt.text(j, i, txt, ha='center', va='center')
plt.tight_layout()
plt.savefig(OUT_CM_PNG, dpi=150)
plt.close()
print(f"Saved confusion matrix -> {OUT_CM_PNG}")

# ---------- tiny HTML summary ----------
if folds.empty:
    table_folds = "<p>No valid folds (training collapsed to <2 classes). With 6 IDs you should not see this — check inputs.</p>"
else:
    table_folds = folds.to_html(index=False)

table_preds = preds.head(30).to_html(index=False) if not preds.empty else "<p>No predictions to show.</p>"

html = f"""
<!doctype html>
<html>
<head><meta charset="utf-8"><title>ML POC — Daily Metrics Classifier</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
h1, h2 {{ margin: 8px 0; }}
small {{ color:#666; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; }}
img {{ max-width: 420px; height: auto; border: 1px solid #eee; border-radius: 8px; padding: 4px; }}
</style>
</head>
<body>
  <h1>ML POC — Daily Metrics Classifier</h1>
  <small>Validation: Leave-One-Participant-Out; Model: StandardScaler + LogisticRegression (liblinear, class_weight=balanced)</small>

  <h2>Per-fold metrics (LOPO)</h2>
  {table_folds}
  <p><b>Overall</b> — Accuracy: {overall_acc if np.isfinite(overall_acc) else 'N/A':.3f} &nbsp;&nbsp; Macro-F1: {overall_f1 if np.isfinite(overall_f1) else 'N/A':.3f} &nbsp;&nbsp; Balanced Acc: {overall_bal if np.isfinite(overall_bal) else 'N/A':.3f}</p>

  <h2>Confusion Matrix</h2>
  <img src="../data/derived/{OUT_CM_PNG.name}" alt="confusion matrix">

  <h2>Sample Predictions (first 30 rows)</h2>
  {table_preds}
</body>
</html>
"""
OUT_HTML.write_text(html, encoding="utf-8")
print(f"Saved HTML -> {OUT_HTML}")