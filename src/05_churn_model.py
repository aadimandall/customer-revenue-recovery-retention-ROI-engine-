"""
05_churn_model.py

Churn-risk modeling layer for the Customer Revenue Recovery & Retention ROI Engine.

This file builds the first predictive layer in the project.

I keep the modeling logic disciplined:
- The churn model answers: who is likely to leave?
- The CLV layer answers: who is worth saving?
- The save-worthiness engine later combines churn risk, CLV, response assumptions, and cost.

The model intentionally avoids target leakage and avoids using final CLV/action fields
as predictors. Customer value is used after scoring to evaluate whether the model ranks
commercially important churn risk near the top of the distribution.

One extra check matters here: I train a sensitivity model that excludes direct
cancellation signals. That lets me see whether the model is learning broader churn
behavior or just leaning on obvious cancellation flags.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


warnings.filterwarnings("ignore")

RANDOM_STATE = 42

PROCESSED_DIR = Path("data/processed")
CLV_DIR = PROCESSED_DIR / "clv_outputs"
OUTPUT_DIR = PROCESSED_DIR / "churn_model_outputs"
MODEL_DIR = Path("models")

CLV_PATH = CLV_DIR / "customer_clv_scores.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "churn_next_period"

# Features are limited to customer state available at the snapshot point.
# Final value/action fields are attached after scoring for business evaluation,
# not used as predictors.
NUMERIC_FEATURES = [
    "age",
    "tenure_months",

    "transaction_count",
    "monthly_revenue",
    "monthly_list_price",
    "monthly_discount_amount",
    "avg_plan_days",
    "max_plan_days",
    "avg_paid_per_day",
    "is_auto_renew",
    "had_cancel",
    "had_discount",

    "latest_payment_plan_days",
    "latest_plan_list_price",
    "latest_actual_amount_paid",
    "latest_is_auto_renew",
    "latest_is_cancel",

    "active_days",
    "total_secs",
    "total_plays",
    "num_unq",
    "completion_rate_proxy",
    "engagement_score",
    "trailing_3mo_engagement_score",
    "trailing_3mo_revenue",
    "trailing_3mo_active_days",
    "engagement_change_pct",
    "activity_change_pct",
    "revenue_change_pct",

    "no_recent_activity_flag",
    "cancellation_signal_flag",
    "major_activity_drop_flag",
]

CATEGORICAL_FEATURES = [
    "city",
    "registered_via",
    "gender",
    "lifecycle_stage",
    "engagement_tier",
    "revenue_tier",
]

# These features are intentionally separated because cancellation behavior can be
# very predictive. The sensitivity model removes them to test whether broader
# engagement and revenue behavior still rank churn risk well.
CANCELLATION_SIGNAL_FEATURES = [
    "had_cancel",
    "latest_is_cancel",
    "cancellation_signal_flag",
]

SENSITIVITY_NUMERIC_FEATURES = [
    col for col in NUMERIC_FEATURES
    if col not in CANCELLATION_SIGNAL_FEATURES
]

# Business columns are carried through after scoring so model performance can be
# translated into churn capture, CLV exposure, and retention planning outputs.
# They are not part of the model feature matrix.
BUSINESS_COLUMNS = [
    "msno",
    "churn_next_period",
    "snapshot_month",

    "lifecycle_stage",
    "engagement_tier",
    "revenue_tier",
    "clv_value_tier",
    "value_based_action_group",
    "retention_budget_tier",

    "monthly_value_baseline",
    "monthly_margin_baseline",
    "annual_revenue_run_rate_proxy",
    "annual_margin_run_rate_proxy",
    "profit_adjusted_clv_proxy",
    "future_churned_clv_proxy",
]

FORBIDDEN_MODEL_FEATURES = [
    "msno",
    "churn_next_period",
    "future_churned_clv_proxy",
    "profit_adjusted_clv_proxy",
    "clv_value_tier",
    "value_based_action_group",
    "retention_budget_tier",
    "actual_future_churn_label",
]


def check_no_leakage_features() -> None:
    # This is a simple guardrail against accidentally adding target, future-value,
    # or final action fields into the model feature list later.
    feature_set = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    forbidden_used = sorted(feature_set.intersection(FORBIDDEN_MODEL_FEATURES))

    if forbidden_used:
        raise ValueError(f"Target leakage risk: forbidden model features used: {forbidden_used}")

    audit = {
        "target": TARGET,
        "numeric_features_main_model": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "cancellation_signal_features": CANCELLATION_SIGNAL_FEATURES,
        "numeric_features_sensitivity_model_without_cancel_signals": SENSITIVITY_NUMERIC_FEATURES,
        "forbidden_model_features": FORBIDDEN_MODEL_FEATURES,
        "note": (
            "Future churned CLV and value/action fields are excluded from model features. "
            "They are used only after scoring for business lift analysis."
        ),
    }

    with open(OUTPUT_DIR / "_model_feature_and_leakage_audit.json", "w") as f:
        json.dump(audit, f, indent=2)


def load_modeling_data() -> pd.DataFrame:
    if not CLV_PATH.exists():
        raise FileNotFoundError(f"Missing {CLV_PATH}. Run src/03_clv_profitability_model.py first.")

    print("\nLoading CLV-scored customer snapshot...")
    df = pd.read_parquet(CLV_PATH)

    needed = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + BUSINESS_COLUMNS)
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for churn modeling: {missing}")

    df = df.copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    # The target is the future churn label used for supervised training.
    # Rows without a known future label cannot be used for model evaluation.

    before = len(df)
    df = df[df[TARGET].notna()].copy()
    removed = before - len(df)

    if removed > 0:
        print(f"Removed {removed:,} rows with missing target.")

    df[TARGET] = df[TARGET].astype(int)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("unknown").astype(str)

    print(f"Loaded modeling rows: {len(df):,}")
    print(f"Future churn rate: {df[TARGET].mean():.2%}")

    return df


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    if scale_numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
    # Ordinal encoding keeps the pipeline compact and works cleanly with the
    # tree-based model. In a production model, I would compare this against
    # one-hot or native categorical handling.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_candidate_models() -> dict[str, dict]:
    # I compare a linear benchmark, a nonlinear main model, and a no-cancel-signal
    # sensitivity model. The goal is not only accuracy, but a defensible ranking model.
    return {
        "logistic_regression_balanced": {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "pipeline": Pipeline(
                steps=[
                    (
                        "preprocessor",
                        build_preprocessor(
                            numeric_features=NUMERIC_FEATURES,
                            categorical_features=CATEGORICAL_FEATURES,
                            scale_numeric=True,
                        ),
                    ),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=700,
                            class_weight="balanced",
                            solver="lbfgs",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "model_note": "Linear benchmark with class weighting.",
            "uses_direct_cancel_signals": True,
        },

        "hist_gradient_boosting": {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "pipeline": Pipeline(
                steps=[
                    (
                        "preprocessor",
                        build_preprocessor(
                            numeric_features=NUMERIC_FEATURES,
                            categorical_features=CATEGORICAL_FEATURES,
                            scale_numeric=False,
                        ),
                    ),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            loss="log_loss",
                            learning_rate=0.06,
                            max_iter=180,
                            max_leaf_nodes=31,
                            min_samples_leaf=40,
                            l2_regularization=0.05,
                            early_stopping=True,
                            validation_fraction=0.12,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "model_note": "Main nonlinear churn-risk model.",
            "uses_direct_cancel_signals": True,
        },

        "hist_gradient_boosting_no_cancel_signals": {
            "numeric_features": SENSITIVITY_NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "pipeline": Pipeline(
                steps=[
                    (
                        "preprocessor",
                        build_preprocessor(
                            numeric_features=SENSITIVITY_NUMERIC_FEATURES,
                            categorical_features=CATEGORICAL_FEATURES,
                            scale_numeric=False,
                        ),
                    ),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            loss="log_loss",
                            learning_rate=0.06,
                            max_iter=180,
                            max_leaf_nodes=31,
                            min_samples_leaf=40,
                            l2_regularization=0.05,
                            early_stopping=True,
                            validation_fraction=0.12,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "model_note": "Sensitivity model excluding direct cancellation signals.",
            "uses_direct_cancel_signals": False,
        },
    }

# The model is mainly used for ranking customers by risk, so ROC-AUC and PR-AUC
# matter more than the default 0.50 classification threshold.
def evaluate_predictions(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.50) -> dict:
    pred = (proba >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier_score": brier_score_loss(y_true, proba),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6)),
        "accuracy_at_0_50": accuracy_score(y_true, pred),
        "precision_at_0_50": precision_score(y_true, pred, zero_division=0),
        "recall_at_0_50": recall_score(y_true, pred, zero_division=0),
        "f1_at_0_50": f1_score(y_true, pred, zero_division=0),
    }


def make_risk_decile_table(y_true: pd.Series, proba: np.ndarray, business_df: pd.DataFrame) -> pd.DataFrame:
    scored = business_df.copy()
    scored["actual_churn"] = y_true.values
    scored["predicted_churn_probability"] = proba

    # Deciles turn model scores into an operating view: if the model is useful,
    # the highest-risk decile should capture more churn and more churned CLV than average.
    scored["risk_decile"] = pd.qcut(
        scored["predicted_churn_probability"].rank(method="first", ascending=False),
        q=10,
        labels=list(range(1, 11)),
    ).astype(int)

    portfolio_churn_rate = scored["actual_churn"].mean()
    total_churners = scored["actual_churn"].sum()
    total_future_churned_clv = scored["future_churned_clv_proxy"].sum()

    decile = (
        scored.groupby("risk_decile", as_index=False)
        .agg(
            customers=("msno", "count"),
            avg_predicted_churn_probability=("predicted_churn_probability", "mean"),
            observed_churn_rate=("actual_churn", "mean"),
            actual_churners=("actual_churn", "sum"),
            profit_adjusted_clv_proxy=("profit_adjusted_clv_proxy", "sum"),
            future_churned_clv_proxy=("future_churned_clv_proxy", "sum"),
            avg_profit_adjusted_clv_proxy=("profit_adjusted_clv_proxy", "mean"),
            avg_monthly_value_baseline=("monthly_value_baseline", "mean"),
        )
        .sort_values("risk_decile")
    )

    decile["lift_vs_portfolio"] = decile["observed_churn_rate"] / portfolio_churn_rate
    decile["churn_capture_share"] = decile["actual_churners"] / total_churners
    decile["cumulative_churn_capture_share"] = decile["churn_capture_share"].cumsum()

    decile["future_churned_clv_share"] = np.where(
        total_future_churned_clv > 0,
        decile["future_churned_clv_proxy"] / total_future_churned_clv,
        0,
    )
    decile["cumulative_future_churned_clv_share"] = decile["future_churned_clv_share"].cumsum()

    return decile


def make_threshold_table(y_true: pd.Series, proba: np.ndarray, business_df: pd.DataFrame) -> pd.DataFrame:
    scored = business_df.copy()
    scored["actual_churn"] = y_true.values
    scored["predicted_churn_probability"] = proba
    scored = scored.sort_values("predicted_churn_probability", ascending=False).reset_index(drop=True)

    portfolio_churn_rate = scored["actual_churn"].mean()
    total_churners = scored["actual_churn"].sum()
    total_future_churned_clv = scored["future_churned_clv_proxy"].sum()

    rows = []

    # Threshold simulation shows the tradeoff between campaign size and churn/value capture.
    # This helps avoid choosing a targeting cutoff just because the model score is high.
    for target_pct in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        n = max(1, int(len(scored) * target_pct))
        top = scored.head(n)

        rows.append(
            {
                "target_population_pct": target_pct,
                "targeted_customers": n,
                "score_threshold": float(top["predicted_churn_probability"].min()),
                "observed_churn_rate_targeted": float(top["actual_churn"].mean()),
                "lift_vs_portfolio": float(top["actual_churn"].mean() / portfolio_churn_rate),
                "churn_capture_rate": float(top["actual_churn"].sum() / total_churners),
                "profit_adjusted_clv_targeted": float(top["profit_adjusted_clv_proxy"].sum()),
                "future_churned_clv_proxy_captured": float(top["future_churned_clv_proxy"].sum()),
                "future_churned_clv_capture_rate": float(
                    top["future_churned_clv_proxy"].sum() / total_future_churned_clv
                    if total_future_churned_clv > 0 else 0
                ),
                "avg_profit_adjusted_clv_targeted": float(top["profit_adjusted_clv_proxy"].mean()),
            }
        )

    return pd.DataFrame(rows)


def make_calibration_table(y_true: pd.Series, proba: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "actual_churn": y_true.values,
            "predicted_churn_probability": proba,
        }
    )

    df["calibration_bin"] = pd.qcut(
        df["predicted_churn_probability"].rank(method="first"),
        q=10,
        labels=list(range(1, 11)),
    ).astype(int)

    calibration = (
        df.groupby("calibration_bin", as_index=False)
        .agg(
            customers=("actual_churn", "count"),
            avg_predicted_churn_probability=("predicted_churn_probability", "mean"),
            observed_churn_rate=("actual_churn", "mean"),
        )
        .sort_values("calibration_bin")
    )

    calibration["calibration_gap"] = (
        calibration["observed_churn_rate"]
        - calibration["avg_predicted_churn_probability"]
    )

    return calibration


def make_scored_customer_file(model: Pipeline, df: pd.DataFrame, model_features: list[str]) -> pd.DataFrame:
    X = df[model_features]
    proba = model.predict_proba(X)[:, 1]

    scored = df[BUSINESS_COLUMNS].copy()
    scored = scored.rename(columns={"churn_next_period": "actual_future_churn_label"})
    scored["predicted_churn_probability"] = proba

    scored["churn_risk_decile"] = pd.qcut(
        scored["predicted_churn_probability"].rank(method="first", ascending=False),
        q=10,
        labels=list(range(1, 11)),
    ).astype(int)

    scored["churn_risk_tier"] = np.select(
        [
            scored["churn_risk_decile"] == 1,
            scored["churn_risk_decile"].between(2, 3),
            scored["churn_risk_decile"].between(4, 6),
            scored["churn_risk_decile"].between(7, 10),
        ],
        [
            "Critical risk",
            "High risk",
            "Medium risk",
            "Low risk",
        ],
        default="Unassigned",
    )
    # This output is intentionally not the final campaign list.
    # Later scripts combine risk with CLV, response assumptions, cost, and A/B test design.
    scored["model_stage_note"] = (
        "Predicted churn risk only. Final targeting comes later from save-worthiness scoring."
    )

    return scored.sort_values("predicted_churn_probability", ascending=False)


def make_segment_score_summary(scored_customers: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scored_customers.groupby(
            [
                "lifecycle_stage",
                "engagement_tier",
                "revenue_tier",
                "clv_value_tier",
                "churn_risk_tier",
                "retention_budget_tier",
            ],
            as_index=False,
        )
        .agg(
            customers=("msno", "count"),
            avg_predicted_churn_probability=("predicted_churn_probability", "mean"),
            actual_future_churn_rate=("actual_future_churn_label", "mean"),
            profit_adjusted_clv_proxy=("profit_adjusted_clv_proxy", "sum"),
            future_churned_clv_proxy=("future_churned_clv_proxy", "sum"),
            avg_profit_adjusted_clv_proxy=("profit_adjusted_clv_proxy", "mean"),
        )
    )

    summary = summary[summary["customers"] >= 250].copy()
    summary = summary.sort_values(
        ["future_churned_clv_proxy", "avg_predicted_churn_probability"],
        ascending=False,
    )

    return summary


def calculate_permutation_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_features: list[str],
) -> pd.DataFrame:
    # Permutation importance can be expensive on a large test set, so I use a
    # fixed sample to keep the script repeatable and practical to rerun.
    sample_size = min(25000, len(X_test))
    sample = X_test.sample(sample_size, random_state=RANDOM_STATE)
    sample_y = y_test.loc[sample.index]

    print(f"\nCalculating permutation importance on {sample_size:,} test rows...")

    importance = permutation_importance(
        model,
        sample,
        sample_y,
        scoring="roc_auc",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    out = pd.DataFrame(
        {
            "feature": model_features,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return out


def validate_model_outputs(
    y_test: pd.Series,
    proba: np.ndarray,
    decile_table: pd.DataFrame,
    scored_customers: pd.DataFrame,
    best_metrics: dict,
) -> pd.DataFrame:
    checks = []

    def add_check(check_name: str, value, passed: bool, notes: str) -> None:
        checks.append(
            {
                "check_name": check_name,
                "status": "PASS" if passed else "FAIL",
                "value": value,
                "notes": notes,
            }
        )

    add_check(
        "predicted_probabilities_in_range",
        f"min={proba.min():.6f}, max={proba.max():.6f}",
        np.all((proba >= 0) & (proba <= 1)),
        "Model probabilities should be between 0 and 1.",
    )

    add_check(
        "roc_auc_above_random",
        round(float(best_metrics["roc_auc"]), 6),
        best_metrics["roc_auc"] > 0.50,
        "ROC-AUC should be above random ranking.",
    )

    add_check(
        "pr_auc_above_base_churn_rate",
        round(float(best_metrics["pr_auc"]), 6),
        best_metrics["pr_auc"] > float(y_test.mean()),
        "PR-AUC should be above the base churn rate.",
    )

    top_decile_lift = float(decile_table.loc[decile_table["risk_decile"] == 1, "lift_vs_portfolio"].iloc[0])
    top_decile_churn_capture = float(decile_table.loc[decile_table["risk_decile"] == 1, "churn_capture_share"].iloc[0])

    add_check(
        "top_decile_lift_above_one",
        round(top_decile_lift, 6),
        top_decile_lift > 1.0,
        "Highest-risk decile should have above-average churn concentration.",
    )

    add_check(
        "top_decile_captures_churners",
        round(top_decile_churn_capture, 6),
        top_decile_churn_capture > 0.10,
        "Top risk decile should capture more than its population share of churners.",
    )

    add_check(
        "scored_customer_output_not_empty",
        len(scored_customers),
        len(scored_customers) > 0,
        "Scored customer output should not be empty.",
    )

    add_check(
        "ten_risk_deciles_created",
        scored_customers["churn_risk_decile"].nunique(),
        scored_customers["churn_risk_decile"].nunique() == 10,
        "Scored customer file should contain ten risk deciles.",
    )

    report = pd.DataFrame(checks)
    report.to_csv(OUTPUT_DIR / "_churn_model_validation_report.csv", index=False)

    with open(OUTPUT_DIR / "_churn_model_validation_report.json", "w") as f:
        json.dump(checks, f, indent=2)

    print("\nValidation report:")
    print(report.to_string(index=False))

    failed = report[report["status"] == "FAIL"]
    if len(failed) > 0:
        raise ValueError("Churn model validation failed. Review outputs before moving forward.")

    return report


def write_model_summary(
    best_model_name: str,
    model_comparison: pd.DataFrame,
    decile_table: pd.DataFrame,
    threshold_table: pd.DataFrame,
    validation_report: pd.DataFrame,
) -> None:
    best = model_comparison.loc[model_comparison["model_name"] == best_model_name].iloc[0]
    top_decile = decile_table.loc[decile_table["risk_decile"] == 1].iloc[0]
    top_20 = threshold_table.loc[np.isclose(threshold_table["target_population_pct"], 0.20)].iloc[0]

    summary = {
        "best_model": best_model_name,
        "best_model_note": str(best["model_note"]),
        "uses_direct_cancel_signals": bool(best["uses_direct_cancel_signals"]),
        "roc_auc": float(best["roc_auc"]),
        "pr_auc": float(best["pr_auc"]),
        "brier_score": float(best["brier_score"]),
        "log_loss": float(best["log_loss"]),
        "test_churn_rate": float(best["test_churn_rate"]),
        "top_decile_observed_churn_rate": float(top_decile["observed_churn_rate"]),
        "top_decile_lift": float(top_decile["lift_vs_portfolio"]),
        "top_decile_churn_capture_share": float(top_decile["churn_capture_share"]),
        "top_decile_future_churned_clv_share": float(top_decile["future_churned_clv_share"]),
        "top_20pct_churn_capture_rate": float(top_20["churn_capture_rate"]),
        "top_20pct_future_churned_clv_capture_rate": float(top_20["future_churned_clv_capture_rate"]),
        "recommended_operating_point": (
            "Top 20% risk segment for planning. Final targeting will use save-worthiness scoring."
        ),
        "validation_status": "PASS" if (validation_report["status"] == "PASS").all() else "FAIL",
    }

    with open(OUTPUT_DIR / "_churn_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Churn Model Summary",
        "",
        f"Best model: {summary['best_model']}",
        f"Model note: {summary['best_model_note']}",
        f"Uses direct cancellation signals: {summary['uses_direct_cancel_signals']}",
        f"ROC-AUC: {summary['roc_auc']:.4f}",
        f"PR-AUC: {summary['pr_auc']:.4f}",
        f"Brier score: {summary['brier_score']:.4f}",
        f"Log loss: {summary['log_loss']:.4f}",
        f"Test churn rate: {summary['test_churn_rate']:.2%}",
        "",
        "Business lift:",
        f"- Top risk decile observed churn rate: {summary['top_decile_observed_churn_rate']:.2%}",
        f"- Top risk decile lift: {summary['top_decile_lift']:.2f}x",
        f"- Top risk decile captures {summary['top_decile_churn_capture_share']:.2%} of churners.",
        f"- Top risk decile captures {summary['top_decile_future_churned_clv_share']:.2%} of future churned CLV proxy.",
        "",
        "Planning operating point:",
        f"- Targeting the top 20% risk segment captures {summary['top_20pct_churn_capture_rate']:.2%} of churners.",
        f"- Targeting the top 20% risk segment captures {summary['top_20pct_future_churned_clv_capture_rate']:.2%} of future churned CLV proxy.",
        "",
        "Interpretation:",
        (
            "This model ranks customers by churn risk. It is not the final retention targeting list. "
            "The next layers combine predicted churn probability with profit-adjusted CLV, save cost, "
            "expected response, and campaign ROI."
        ),
        "",
        "Leakage control:",
        (
            "Future-churned value fields and final value/action fields were excluded from model features. "
            "They are used only after scoring for retrospective business lift analysis."
        ),
    ]

    with open(OUTPUT_DIR / "_executive_churn_model_summary.md", "w") as f:
        f.write("\n".join(lines))

    print(f"Saved {OUTPUT_DIR / '_executive_churn_model_summary.md'}")
    print(f"Saved {OUTPUT_DIR / '_churn_model_summary.json'}")


def main() -> None:
    print("\nTraining churn-risk model...")

    check_no_leakage_features()
    df = load_modeling_data()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    business = df[BUSINESS_COLUMNS].copy()

    X_train, X_test, y_train, y_test, business_train, business_test = train_test_split(
        X,
        y,
        business,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Train churn rate: {y_train.mean():.2%}")
    print(f"Test churn rate: {y_test.mean():.2%}")

    models = build_candidate_models()
    fitted_models = {}
    comparison_rows = []

    for model_name, model_config in models.items():
        print(f"\nFitting {model_name}...")

        model = model_config["pipeline"]
        model_features = model_config["numeric_features"] + model_config["categorical_features"]

        model.fit(X_train[model_features], y_train)

        proba = model.predict_proba(X_test[model_features])[:, 1]
        metrics = evaluate_predictions(y_test, proba)
        decile_for_model = make_risk_decile_table(y_test, proba, business_test)

        top_decile = decile_for_model.loc[decile_for_model["risk_decile"] == 1].iloc[0]

        row = {
            "model_name": model_name,
            "model_note": model_config["model_note"],
            "uses_direct_cancel_signals": model_config["uses_direct_cancel_signals"],
            "feature_count": len(model_features),
            "test_churn_rate": float(y_test.mean()),
            **metrics,
            "top_decile_lift": float(top_decile["lift_vs_portfolio"]),
            "top_decile_churn_capture_share": float(top_decile["churn_capture_share"]),
            "top_decile_future_churned_clv_share": float(top_decile["future_churned_clv_share"]),
        }

        comparison_rows.append(row)
        fitted_models[model_name] = model

        print(
            f"{model_name}: ROC-AUC={metrics['roc_auc']:.4f} | "
            f"PR-AUC={metrics['pr_auc']:.4f} | "
            f"Top decile lift={top_decile['lift_vs_portfolio']:.2f}x"
        )

    # Model selection prioritizes PR-AUC first because churn is an imbalanced
    # targeting problem. ROC-AUC and top-decile lift are used as supporting checks.
    model_comparison = pd.DataFrame(comparison_rows).sort_values(
        ["pr_auc", "roc_auc", "top_decile_lift"],
        ascending=False,
    )

    best_model_name = model_comparison.iloc[0]["model_name"]
    best_model = fitted_models[best_model_name]
    best_model_config = models[best_model_name]
    best_model_features = best_model_config["numeric_features"] + best_model_config["categorical_features"]

    print(f"\nBest model selected: {best_model_name}")

    best_proba_test = best_model.predict_proba(X_test[best_model_features])[:, 1]
    best_metrics = evaluate_predictions(y_test, best_proba_test)

    decile_table = make_risk_decile_table(y_test, best_proba_test, business_test)
    threshold_table = make_threshold_table(y_test, best_proba_test, business_test)
    calibration_table = make_calibration_table(y_test, best_proba_test)
    scored_customers = make_scored_customer_file(best_model, df, best_model_features)
    segment_score_summary = make_segment_score_summary(scored_customers)

    feature_importance = calculate_permutation_importance(
        best_model,
        X_test[best_model_features],
        y_test,
        best_model_features,
    )

    validation_report = validate_model_outputs(
        y_test=y_test,
        proba=best_proba_test,
        decile_table=decile_table,
        scored_customers=scored_customers,
        best_metrics=best_metrics,
    )

    model_comparison.to_csv(OUTPUT_DIR / "00_model_comparison.csv", index=False)
    decile_table.to_csv(OUTPUT_DIR / "01_risk_decile_business_summary.csv", index=False)
    threshold_table.to_csv(OUTPUT_DIR / "02_targeting_threshold_simulation.csv", index=False)
    calibration_table.to_csv(OUTPUT_DIR / "03_calibration_by_score_bin.csv", index=False)
    feature_importance.to_csv(OUTPUT_DIR / "04_feature_importance_permutation.csv", index=False)
    segment_score_summary.to_csv(OUTPUT_DIR / "05_segment_score_summary.csv", index=False)

    scored_customers.to_parquet(OUTPUT_DIR / "churn_scored_customers.parquet", index=False)

    scored_customers.head(100000).to_csv(
        OUTPUT_DIR / "tableau_churn_scored_customer_sample.csv",
        index=False,
    )

    # Save the full preprocessing + model pipeline so scoring can be reproduced
    # without manually recreating feature transformations.
    joblib.dump(best_model, MODEL_DIR / "churn_model_pipeline.joblib")

    training_manifest = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "best_model": str(best_model_name),
        "best_model_note": models[best_model_name]["model_note"],
        "best_model_features": best_model_features,
        "best_model_uses_direct_cancel_signals": bool(models[best_model_name]["uses_direct_cancel_signals"]),
        "sensitivity_test": "Included a challenger model that excludes direct cancellation signals.",
        "model_selection_sort": ["pr_auc", "roc_auc", "top_decile_lift"],
        "numeric_features_main_model": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_from_features_but_used_for_business_evaluation": [
            "future_churned_clv_proxy",
            "profit_adjusted_clv_proxy",
            "clv_value_tier",
            "value_based_action_group",
            "retention_budget_tier",
        ],
    }

    with open(OUTPUT_DIR / "_model_training_manifest.json", "w") as f:
        json.dump(training_manifest, f, indent=2)

    write_model_summary(
        best_model_name=best_model_name,
        model_comparison=model_comparison,
        decile_table=decile_table,
        threshold_table=threshold_table,
        validation_report=validation_report,
    )

    print("\nSaved outputs:")
    print(OUTPUT_DIR / "00_model_comparison.csv")
    print(OUTPUT_DIR / "01_risk_decile_business_summary.csv")
    print(OUTPUT_DIR / "02_targeting_threshold_simulation.csv")
    print(OUTPUT_DIR / "03_calibration_by_score_bin.csv")
    print(OUTPUT_DIR / "04_feature_importance_permutation.csv")
    print(OUTPUT_DIR / "05_segment_score_summary.csv")
    print(OUTPUT_DIR / "churn_scored_customers.parquet")
    print(OUTPUT_DIR / "tableau_churn_scored_customer_sample.csv")
    print(MODEL_DIR / "churn_model_pipeline.joblib")

    print("\n05_churn_model.py complete.")


if __name__ == "__main__":
    main()
