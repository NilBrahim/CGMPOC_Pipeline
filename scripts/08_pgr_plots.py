# scripts/08_pgr_plots.py
# Purpose: quick visuals for the PGR POC model
# Inputs: data/derived/pgr_folds.csv, pgr_predictions.csv, pgr_coefficients.csv
# Outputs: plots into reports/: y_true_vs_pred.png, residuals_by_pid.png, coef_bar.png
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import DATA_DERIVED, REPORTS, ID_COL

FOLDS = DATA_DERIVED / "pgr_folds.csv"
PREDS = DATA_DERIVED / "pgr_predictions.csv"
COEFS = DATA_DERIVED / "pgr_coefficients.csv"

OUT_SCATTER = REPORTS / "y_true_vs_pred.png"
OUT_RESID   = REPORTS / "residuals_by_pid.png"
OUT_COEF    = REPORTS / "coef_bar.png"

def main():
    print("Reading:", FOLDS)
    print("Reading:", PREDS)
    print("Reading:", COEFS)
    folds = pd.read_csv(FOLDS)
    preds = pd.read_csv(PREDS)
    coefs = pd.read_csv(COEFS)

    # --- basic metrics summary ---
    if not folds.empty:
        print("\nPer-fold metrics (RMSE/MAE/R2):")
        print(folds.to_string(index=False))
        print("\nOverall means:",
              f"RMSE={folds['rmse'].mean():.2f}",
              f"MAE={folds['mae'].mean():.2f}",
              f"R²={folds['r2'].mean():.3f}")
    else:
        print("[warn] folds file is empty")

    # --- scatter: y_true vs y_pred ---
    if not preds.empty:
        plt.figure(figsize=(5.5,5))
        x = preds["y_true"].values
        y = preds["y_pred"].values
        plt.scatter(x, y, s=14, alpha=0.6)
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        plt.plot(lim, lim, linewidth=1)
        plt.xlabel("True Δpeak (mg/dL)")
        plt.ylabel("Pred Δpeak (mg/dL)")
        plt.title("PGR — True vs Predicted")
        plt.tight_layout()
        plt.savefig(OUT_SCATTER, dpi=150)
        plt.close()
        print("Saved:", OUT_SCATTER)

        # residuals by participant
        if ID_COL in preds.columns:
            preds["resid"] = preds["y_pred"] - preds["y_true"]
            g = preds.groupby(ID_COL)["resid"].agg(["mean","median","std","count"]).reset_index()
            print("\nResiduals by participant:")
            print(g.to_string(index=False))

            plt.figure(figsize=(6.5,4))
            order = g.sort_values("mean")[ID_COL].astype(int).tolist()
            plt.bar([str(p) for p in order],
                    g.set_index(ID_COL).loc[order, "mean"].values)
            plt.xlabel("Participant")
            plt.ylabel("Residual mean (pred - true)")
            plt.title("Residuals by participant")
            plt.tight_layout()
            plt.savefig(OUT_RESID, dpi=150)
            plt.close()
            print("Saved:", OUT_RESID)
    else:
        print("[warn] preds file is empty")

    # --- coefficients (average across folds) ---
    if not coefs.empty:
        # drop columns that aren't features
        drop_cols = {"left_out_pid", "intercept"}
        feat_cols = [c for c in coefs.columns if c not in drop_cols]
        coef_mean = coefs[feat_cols].mean(axis=0).sort_values(ascending=True)
        plt.figure(figsize=(7, max(3, 0.4*len(coef_mean))))
        plt.barh(coef_mean.index, coef_mean.values)
        plt.title("Average coefficients (original units)\n(+ => larger Δpeak, − => attenuated)")
        plt.tight_layout()
        plt.savefig(OUT_COEF, dpi=150)
        plt.close()
        print("Saved:", OUT_COEF)
        print("\nTop drivers (absolute magnitude):")
        print(coef_mean.reindex(coef_mean.abs().sort_values(ascending=False).index).head(10))
    else:
        print("[warn] coefficients file is empty")

if __name__ == "__main__":
    main()