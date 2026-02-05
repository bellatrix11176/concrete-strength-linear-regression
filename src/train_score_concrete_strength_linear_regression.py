from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# =============================================================================
# Concrete Strength — Linear Regression (Train + Prune + Score + Evidence Locker)
# =============================================================================

LABEL_COL = "CompressiveStrength"
ID_COL = "SlabID"
ALPHA = 0.05

# -----------------------------
# Repo-friendly paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../Linear Regression
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "concrete_mix_strength_train.csv"
SCORE_PATH = DATA_DIR / "concrete_mix_strength_score.csv"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    # -----------------------------
    # Load data
    # -----------------------------
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")
    if not SCORE_PATH.exists():
        raise FileNotFoundError(f"Scoring file not found: {SCORE_PATH}")

    train = pd.read_csv(TRAIN_PATH)
    score = pd.read_csv(SCORE_PATH)

    if LABEL_COL not in train.columns:
        raise ValueError(f"Training file is missing label column: {LABEL_COL}")
    if ID_COL not in train.columns or ID_COL not in score.columns:
        raise ValueError(f"Expected ID column '{ID_COL}' in both train and score.")

    feature_cols = [c for c in train.columns if c not in [LABEL_COL, ID_COL]]

    missing_in_score = sorted(set(feature_cols) - set(score.columns))
    if missing_in_score:
        raise ValueError(f"Scoring data is missing these feature columns: {missing_in_score}")

    # -----------------------------
    # Coerce numeric columns (defensive)
    # -----------------------------
    for col in feature_cols + [LABEL_COL]:
        train[col] = pd.to_numeric(train[col], errors="coerce")
    for col in feature_cols:
        score[col] = pd.to_numeric(score[col], errors="coerce")

    # Keep only rows with a label for training
    train = train.dropna(subset=[LABEL_COL]).copy()

    # -----------------------------
    # Training ranges + range-filter scoring
    # -----------------------------
    ranges = train[feature_cols].agg(["min", "max"]).T.rename(columns={"min": "train_min", "max": "train_max"})
    ranges.index.name = "feature"
    ranges.to_csv(OUT_DIR / "training_feature_ranges.csv")

    out_of_range_summary = []
    in_range_mask = pd.Series(True, index=score.index)

    for col in feature_cols:
        col_min = float(ranges.loc[col, "train_min"])
        col_max = float(ranges.loc[col, "train_max"])

        below = score[col] < col_min
        above = score[col] > col_max

        out_of_range_summary.append(
            {
                "feature": col,
                "train_min": col_min,
                "train_max": col_max,
                "score_below_min_count": int(below.sum()),
                "score_above_max_count": int(above.sum()),
                "score_out_of_range_count": int((below | above).sum()),
            }
        )

        # Treat NaN as out-of-range
        in_range_mask &= score[col].between(col_min, col_max, inclusive="both") & score[col].notna()

    pd.DataFrame(out_of_range_summary).sort_values(
        by="score_out_of_range_count", ascending=False
    ).to_csv(OUT_DIR / "scoring_out_of_range_summary.csv", index=False)

    score_in_range = score[in_range_mask].copy()
    score_removed = score[~in_range_mask].copy()
    if not score_removed.empty:
        score_removed.to_csv(OUT_DIR / "scoring_rows_removed_out_of_range.csv", index=False)

    # -----------------------------
    # Fit model with all predictors (p-values)
    # -----------------------------
    X_all = sm.add_constant(train[feature_cols], has_constant="add")
    y = train[LABEL_COL].astype(float)

    valid_train_mask = X_all.notna().all(axis=1) & y.notna()
    X_all = X_all.loc[valid_train_mask]
    y = y.loc[valid_train_mask]

    model_all = sm.OLS(y, X_all).fit()
    pvals_all = model_all.pvalues.drop(labels=["const"], errors="ignore").sort_values()
    pvals_all.to_csv(OUT_DIR / "all_predictors_pvalues.csv", header=["p_value"])

    sig = pvals_all[pvals_all <= ALPHA].index.tolist()
    nonsig = pvals_all[pvals_all > ALPHA].index.tolist()

    # -----------------------------
    # Refit final model (significant predictors only)
    # -----------------------------
    final_feats = sig if sig else feature_cols

    X_final = sm.add_constant(train[final_feats], has_constant="add")
    y_final = train[LABEL_COL].astype(float)

    valid_train_mask2 = X_final.notna().all(axis=1) & y_final.notna()
    X_final = X_final.loc[valid_train_mask2]
    y_final = y_final.loc[valid_train_mask2]

    model_final = sm.OLS(y_final, X_final).fit()

    final_pvals = model_final.pvalues.drop(labels=["const"], errors="ignore").sort_values()
    final_pvals.to_csv(OUT_DIR / "final_model_pvalues.csv", header=["p_value"])

    weakest_predictor = (
        model_final.pvalues.drop(labels=["const"], errors="ignore")
        .sort_values(ascending=False)
        .index[0]
        if len(final_feats) > 0
        else "None"
    )

    # -----------------------------
    # Coefficients + standardized coefficients
    # -----------------------------
    coefs = model_final.params.copy()
    coef_df = pd.DataFrame(
        {
            "feature": coefs.index,
            "coefficient": coefs.values,
            "p_value": model_final.pvalues.reindex(coefs.index).values,
        }
    )

    std_y = float(y_final.std(ddof=0))
    std_x = train.loc[valid_train_mask2, final_feats].std(ddof=0).to_dict()

    std_betas = []
    for feat in coef_df["feature"]:
        if feat == "const" or std_y == 0:
            std_betas.append(np.nan)
        else:
            std_betas.append(float(coefs[feat]) * (float(std_x[feat]) / std_y))

    coef_df["standardized_coefficient"] = std_betas
    coef_df.to_csv(OUT_DIR / "model_coefficients.csv", index=False)

    with open(OUT_DIR / "linear_regression_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(model_final.summary().as_text())

    # -----------------------------
    # Training metrics
    # -----------------------------
    y_pred_train = model_final.predict(X_final).astype(float).to_numpy()
    y_true_train = y_final.to_numpy()
    residuals = y_true_train - y_pred_train

    r2 = float(model_final.rsquared)

    pd.DataFrame(
        {
            "metric": ["r_squared", "rmse", "mae", "n_train_used", "n_features_final"],
            "value": [
                r2,
                rmse(y_true_train, y_pred_train),
                mae(y_true_train, y_pred_train),
                int(len(y_true_train)),
                int(len(final_feats)),
            ],
        }
    ).to_csv(OUT_DIR / "regression_metrics.csv", index=False)

    # -----------------------------
    # Score the scoring dataset (in-range only)
    # -----------------------------
    X_score = sm.add_constant(score_in_range[final_feats], has_constant="add")
    valid_score_mask = X_score.notna().all(axis=1)

    score_in_range_valid = score_in_range.loc[valid_score_mask].copy()
    X_score_valid = X_score.loc[valid_score_mask]

    preds_score = model_final.predict(X_score_valid)

    pred_df = pd.DataFrame(
        {
            ID_COL: score_in_range_valid[ID_COL].values,
            "PredictedCompressiveStrength": preds_score.values,
        }
    )
    pred_df.to_csv(OUT_DIR / "concrete_strength_predictions.csv", index=False)

    top10 = pred_df.sort_values("PredictedCompressiveStrength", ascending=False).head(10)
    top10.to_csv(OUT_DIR / "top10_predicted_strength.csv", index=False)

    strongest_row = pred_df.loc[pred_df["PredictedCompressiveStrength"].idxmax()]
    strongest_slab_id = int(strongest_row[ID_COL])
    strongest_pred = float(strongest_row["PredictedCompressiveStrength"])

    # -----------------------------
    # Visuals
    # -----------------------------
    plt.figure()
    plt.hist(y_true_train, bins=30)
    plt.title("Training Distribution: CompressiveStrength")
    plt.xlabel("CompressiveStrength")
    plt.ylabel("Count")
    save_fig(OUT_DIR / "distribution_compressive_strength.png")

    corr_cols = list(final_feats) + [LABEL_COL]
    corr = train[corr_cols].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.title("Correlation Heatmap (Final Predictors + Label)")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()
    save_fig(OUT_DIR / "correlation_heatmap.png")

    plt.figure()
    plt.scatter(y_true_train, y_pred_train, s=12)
    lo = float(min(y_true_train.min(), y_pred_train.min()))
    hi = float(max(y_true_train.max(), y_pred_train.max()))
    plt.plot([lo, hi], [lo, hi])
    plt.title("Actual vs Predicted (Training)")
    plt.xlabel("Actual CompressiveStrength")
    plt.ylabel("Predicted CompressiveStrength")
    save_fig(OUT_DIR / "actual_vs_predicted.png")

    plt.figure()
    plt.scatter(y_pred_train, residuals, s=12)
    plt.axhline(0.0)
    plt.title("Residuals vs Predicted (Training)")
    plt.xlabel("Predicted CompressiveStrength")
    plt.ylabel("Residual (Actual - Predicted)")
    save_fig(OUT_DIR / "residuals_vs_predicted.png")

    plt.figure()
    plt.hist(residuals, bins=30)
    plt.title("Residual Distribution (Training)")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    save_fig(OUT_DIR / "residual_distribution.png")

    # -----------------------------
    # Run summary
    # -----------------------------
    missing_predictor_rows = int((~valid_score_mask).sum())

    print("\n=== OUTPUT EVIDENCE LOCKER ===")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"Training file: {TRAIN_PATH}")
    print(f"Scoring file:  {SCORE_PATH}")
    print(f"\nRemoved scoring rows (out of range): {len(score_removed)}")
    print(f"Also removed scoring rows (missing predictors): {missing_predictor_rows}")
    print(f"Strongest predicted SlabID: {strongest_slab_id} (pred={strongest_pred:.4f})")
    print(f"\nFinal predictors (alpha={ALPHA}): {final_feats}")
    print(f"Non-significant (p > {ALPHA}) before pruning: {nonsig if nonsig else 'None'}")
    print(f"Weakest predictor in final model (highest p-value among kept): {weakest_predictor}")
    print(f"\nR-squared: {r2:.4f} => {r2 * 100:.1f}%")

    print("\nFiles written to output/:")
    print(" - training_feature_ranges.csv")
    print(" - scoring_out_of_range_summary.csv")
    if not score_removed.empty:
        print(" - scoring_rows_removed_out_of_range.csv")
    print(" - all_predictors_pvalues.csv")
    print(" - final_model_pvalues.csv")
    print(" - model_coefficients.csv")
    print(" - regression_metrics.csv")
    print(" - concrete_strength_predictions.csv")
    print(" - top10_predicted_strength.csv")
    print(" - linear_regression_model_summary.txt")
    print(" - distribution_compressive_strength.png")
    print(" - correlation_heatmap.png")
    print(" - actual_vs_predicted.png")
    print(" - residuals_vs_predicted.png")
    print(" - residual_distribution.png")


if __name__ == "__main__":
    main()
