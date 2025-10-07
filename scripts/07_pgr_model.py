# scripts/07_pgr_model.py
# Goal: Interpretable PGR model — predict delta_peak (mg/dL) from multimodal features.
# - Data: data/derived/meal_outcomes.csv (from step 06)
# - Model: StandardScaler + Ridge (linear, interpretable)
# - CV: Leave-One-Participant-Out (LOPO) by Person_ID
# - Saves: per-fold metrics, per-meal predictions, coefficients, and an HTML summary.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import DATA_DERIVED, REPORTS, ID_COL

IN_CSV   = DATA_DERIVED / "meal_outcomes.csv"
OUT_PRED = DATA_DERIVED / "pgr_predictions.csv"
OUT_FOLD = DATA_DERIVED / "pgr_folds.csv"
OUT_COEF = DATA_DERIVED / "pgr_coefficients.csv"
OUT_HTML = REPORTS / "PGR_POC_summary.html"

TARGET = "delta_peak"

# Candidate features (safe if missing)
FEATS = [
    "carbs_g","protein_g","fat_g","fiber_g","calories",
    "baseline_mgdl","baseline_sd",
    "gap_prev_meal_hr","hour",
    "steps_prev_120m","hr_mean_prev_60m",
]

def recover_original_space_coefs(pipeline, X_df: pd.DataFrame):
    """
    Convert pipeline (StandardScaler -> Ridge) coefficients back to original feature units.
    Returns coef Series aligned to X_df.columns and an intercept in original units.
    """
    scaler = pipeline.named_steps["scaler"]
    ridge  = pipeline.named_steps["model"]  # Ridge
    w_std  = ridge.coef_.astype(float)      # weights in standardized space
    b_std  = float(ridge.intercept_)

    scale = scaler.scale_
    mean  = scaler.mean_

    w_orig = w_std / scale
    b_orig = b_std - np.sum((w_std * mean) / scale)
    return pd.Series(w_orig, index=X_df.columns), float(b_orig)

def main():
    print("Reading:", IN_CSV)
    df = pd.read_csv(IN_CSV, parse_dates=["meal_time"])

    # Keep only rows with target present and nonnegative (delta_peak should be >= 0)
    df = df[pd.notna(df[TARGET])]
    df = df[df[TARGET] >= 0].copy()

    # Build feature frame, coerce numeric
    X = df[[c for c in FEATS if c in df.columns]].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Simple imputations: 0 for activity (often sparse), mean for nutrition/baseline
    activity_cols = [c for c in ["steps_prev_120m","hr_mean_prev_60m"] if c in X.columns]
    for c in activity_cols:
        X[c] = X[c].fillna(0.0)
    for c in X.columns:
        if c not in activity_cols:
            X[c] = X[c].fillna(X[c].median())

    y = pd.to_numeric(df[TARGET], errors="coerce").values
    pids = df[ID_COL].values

    # Sanity
    if X.empty or len(df) == 0:
        raise ValueError("No rows available for modeling. Check meal_outcomes.csv")

    # Model pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model",  Ridge(alpha=1.0, random_state=0))
    ])

    # LOPO CV
    unique_pids = np.unique(pids)
    fold_rows = []
    pred_rows = []
    coef_rows = []

    for pid in unique_pids:
        test_idx  = (pids == pid)
        train_idx = ~test_idx

        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test,  y_test  = X.iloc[test_idx],  y[test_idx]

        if len(X_test) == 0 or len(X_train) < 5:
            print(f"[warn] Skipping pid={pid}: insufficient data.")
            continue

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae  = float(mean_absolute_error(y_test, y_pred))
        r2   = float(r2_score(y_test, y_pred))

        fold_rows.append({
            "left_out_pid": int(pid),
            "n_test_meals": int(test_idx.sum()),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        })

        # store predictions
        pred_rows.append(pd.DataFrame({
            "Person_ID": df.loc[test_idx, ID_COL].astype(int).values,
            "meal_time": df.loc[test_idx, "meal_time"].dt.strftime("%Y-%m-%d %H:%M").values,
            "y_true":    y_test,
            "y_pred":    y_pred,
        }))

        # recover coefficients in original units
        coefs, intercept = recover_original_space_coefs(pipe, X_train)
        row = coefs.to_dict()
        row.update({"left_out_pid": int(pid), "intercept": intercept})
        coef_rows.append(row)

    # Concatenate outputs
    folds = pd.DataFrame(fold_rows).sort_values("left_out_pid")
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    coefs = pd.DataFrame(coef_rows).sort_values("left_out_pid") if coef_rows else pd.DataFrame()

    # Overall metrics
    if not folds.empty:
        overall = {
            "rmse_mean": folds["rmse"].mean(),
            "mae_mean":  folds["mae"].mean(),
            "r2_mean":   folds["r2"].mean()
        }
    else:
        overall = {"rmse_mean": np.nan, "mae_mean": np.nan, "r2_mean": np.nan}

    # Save artifacts
    folds.to_csv(OUT_FOLD, index=False)
    preds.to_csv(OUT_PRED, index=False)
    coefs.to_csv(OUT_COEF, index=False)
    print(f"Saved per-fold metrics -> {OUT_FOLD}")
    print(f"Saved predictions      -> {OUT_PRED}")
    print(f"Saved coefficients     -> {OUT_COEF}")

    # HTML summary (mini)
    coef_preview = (coefs.set_index("left_out_pid")
                         .drop(columns=["intercept"], errors="ignore")
                         .mean(axis=0, skipna=True)
                         .sort_values(ascending=False)
                         .to_frame("avg_coef")
                         .to_html(border=0))
    folds_html = folds.to_html(index=False, border=0) if not folds.empty else "<p>No folds.</p>"
    preds_html = preds.head(30).to_html(index=False, border=0) if not preds.empty else "<p>No predictions.</p>"

    html = f"""
<!doctype html>
<html>
<head><meta charset="utf-8"><title>PGR POC — Ridge (interpretable)</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
h1, h2 {{ margin: 8px 0; }}
small {{ color:#666; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; }}
</style>
</head>
<body>
  <h1>PGR POC — Predicting ΔPeak (mg/dL)</h1>
  <small>Model: StandardScaler + Ridge (LOPO by participant). Target: delta_peak in first 120 min.</small>

  <h2>Per-fold metrics (LOPO)</h2>
  {folds_html}
  <p><b>Overall</b> — RMSE: {overall['rmse_mean']:.2f} &nbsp; MAE: {overall['mae_mean']:.2f} &nbsp; R²: {overall['r2_mean']:.3f}</p>

  <h2>Average coefficients (original units)</h2>
  <p>Higher positive values → larger postprandial rise; negative values → attenuate rise.</p>
  {coef_preview}

  <h2>Sample predictions (first 30)</h2>
  {preds_html}
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Saved HTML -> {OUT_HTML}")

if __name__ == "__main__":
    main()