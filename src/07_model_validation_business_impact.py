"""
07_model_validation_business_impact.py

Business validation layer for the Customer Revenue Recovery & Retention ROI Engine.

The churn model is not valuable because it produces a probability.
It is valuable only if the ranking helps the business make better retention decisions.

This script checks that directly.

It answers:
1. Does the model rank churn risk better than random?
2. Does the highest-risk population concentrate actual churners?
3. Does the highest-risk population also concentrate churned customer value?
4. Is the model calibrated well enough to support planning?
5. Is the model too dependent on obvious cancellation flags?
6. Which operating threshold should leadership use before save-worthiness scoring?
7. What should be trusted, what should be treated cautiously, and what goes into the next layer?

Important distinction:
This is still model validation, not final targeting. The final retention decision comes later
when predicted churn probability is combined with CLV, expected response, gross margin,
intervention cost, and ROI.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed")
CHURN_DIR = PROCESSED_DIR / "churn_model_outputs"
OUTPUT_DIR = PROCESSED_DIR / "model_validation_outputs"
SQL_DIR = Path("sql")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SQL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COMPARISON_PATH = CHURN_DIR / "00_model_comparison.csv"
DECILE_PATH = CHURN_DIR / "01_risk_decile_business_summary.csv"
THRESHOLD_PATH = CHURN_DIR / "02_targeting_threshold_simulation.csv"
CALIBRATION_PATH = CHURN_DIR / "03_calibration_by_score_bin.csv"
FEATURE_IMPORTANCE_PATH = CHURN_DIR / "04_feature_importance_permutation.csv"
SEGMENT_SCORE_PATH = CHURN_DIR / "05_segment_score_summary.csv"
SCORED_CUSTOMERS_PATH = CHURN_DIR / "churn_scored_customers.parquet"
CHURN_SUMMARY_PATH = CHURN_DIR / "_churn_model_summary.json"
FEATURE_AUDIT_PATH = CHURN_DIR / "_model_feature_and_leakage_audit.json"
TRAINING_MANIFEST_PATH = CHURN_DIR / "_model_training_manifest.json"

# Segment readouts below this size can be noisy, so I filter small groups
# before turning them into business recommendations.
MIN_SEGMENT_SIZE = 500


REQUIRED_SCORED_COLUMNS = [
    "msno",
    "actual_future_churn_label",
    "predicted_churn_probability",
    "churn_risk_decile",
    "churn_risk_tier",
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


def to_builtin(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    return to_builtin(obj)


def read_json_if_exists(path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def require_file(path, upstream_script):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run {upstream_script} first.")


def load_inputs():
    require_file(MODEL_COMPARISON_PATH, "src/05_churn_model.py")
    require_file(DECILE_PATH, "src/05_churn_model.py")
    require_file(THRESHOLD_PATH, "src/05_churn_model.py")
    require_file(CALIBRATION_PATH, "src/05_churn_model.py")
    require_file(SCORED_CUSTOMERS_PATH, "src/05_churn_model.py")

    # This layer does not retrain the model. It reads the modeling outputs and
    # translates them into business validation, governance, and planning views.
    model_comparison = pd.read_csv(MODEL_COMPARISON_PATH)
    decile = pd.read_csv(DECILE_PATH)
    threshold = pd.read_csv(THRESHOLD_PATH)
    calibration = pd.read_csv(CALIBRATION_PATH)

    feature_importance = (
        pd.read_csv(FEATURE_IMPORTANCE_PATH)
        if FEATURE_IMPORTANCE_PATH.exists()
        else pd.DataFrame()
    )

    segment_score = (
        pd.read_csv(SEGMENT_SCORE_PATH)
        if SEGMENT_SCORE_PATH.exists()
        else pd.DataFrame()
    )

    scored = pd.read_parquet(SCORED_CUSTOMERS_PATH)

    missing = sorted(set(REQUIRED_SCORED_COLUMNS) - set(scored.columns))
    if missing:
        raise ValueError(f"Scored customer file is missing required columns: {missing}")

    churn_summary = read_json_if_exists(CHURN_SUMMARY_PATH)
    feature_audit = read_json_if_exists(FEATURE_AUDIT_PATH)
    training_manifest = read_json_if_exists(TRAINING_MANIFEST_PATH)

    return {
        "model_comparison": model_comparison,
        "decile": decile,
        "threshold": threshold,
        "calibration": calibration,
        "feature_importance": feature_importance,
        "segment_score": segment_score,
        "scored": scored,
        "churn_summary": churn_summary,
        "feature_audit": feature_audit,
        "training_manifest": training_manifest,
    }


def pick_best_model(model_comparison):
    # PR-AUC is prioritized because churn is an imbalanced targeting problem.
    # ROC-AUC and top-decile lift are supporting ranking-quality checks.
    sorted_models = model_comparison.sort_values(
        ["pr_auc", "roc_auc", "top_decile_lift"],
        ascending=False,
    ).reset_index(drop=True)

    return sorted_models.iloc[0], sorted_models


def business_grade(roc_auc, pr_auc, base_churn_rate, top_decile_lift, top20_value_capture):
    # PR-AUC is compared against the base churn rate so the score reflects
    # improvement over the natural churn prevalence, not just an isolated metric.
    pr_auc_multiple = pr_auc / base_churn_rate if base_churn_rate > 0 else np.nan

    if (
        roc_auc >= 0.80
        and pr_auc_multiple >= 4
        and top_decile_lift >= 5
        and top20_value_capture >= 0.75
    ):
        return "Excellent business ranking model"

    if (
        roc_auc >= 0.75
        and pr_auc_multiple >= 3
        and top_decile_lift >= 4
        and top20_value_capture >= 0.60
    ):
        return "Strong business ranking model"

    if (
        roc_auc >= 0.70
        and pr_auc_multiple >= 2
        and top_decile_lift >= 2.5
        and top20_value_capture >= 0.45
    ):
        return "Usable business ranking model"

    return "Needs caution before broad deployment"


def make_validation_kpi_summary(inputs):
    model_comparison = inputs["model_comparison"]
    decile = inputs["decile"]
    threshold = inputs["threshold"]

    best, _ = pick_best_model(model_comparison)

    # The top decile is the key operating check: a useful ranking model should
    # concentrate churners and churned value near the top of the score distribution.
    top_decile = decile.loc[decile["risk_decile"] == 1].iloc[0]

    top20 = threshold.loc[
        np.isclose(threshold["target_population_pct"], 0.20)
    ].iloc[0]

    base_churn_rate = float(best["test_churn_rate"])
    pr_auc_multiple = float(best["pr_auc"] / base_churn_rate) if base_churn_rate > 0 else np.nan

    grade = business_grade(
        roc_auc=float(best["roc_auc"]),
        pr_auc=float(best["pr_auc"]),
        base_churn_rate=base_churn_rate,
        top_decile_lift=float(top_decile["lift_vs_portfolio"]),
        top20_value_capture=float(top20["future_churned_clv_capture_rate"]),
    )

    row = {
        "best_model": best["model_name"],
        "model_note": best.get("model_note", ""),
        "uses_direct_cancel_signals": bool(best.get("uses_direct_cancel_signals", False)),
        "test_churn_rate": base_churn_rate,
        "roc_auc": float(best["roc_auc"]),
        "pr_auc": float(best["pr_auc"]),
        "pr_auc_multiple_vs_base_churn": pr_auc_multiple,
        "brier_score": float(best["brier_score"]),
        "log_loss": float(best["log_loss"]),
        "top_decile_observed_churn_rate": float(top_decile["observed_churn_rate"]),
        "top_decile_lift": float(top_decile["lift_vs_portfolio"]),
        "top_decile_churn_capture_share": float(top_decile["churn_capture_share"]),
        "top_decile_future_churned_clv_share": float(top_decile["future_churned_clv_share"]),
        "top20_churn_capture_rate": float(top20["churn_capture_rate"]),
        "top20_future_churned_clv_capture_rate": float(top20["future_churned_clv_capture_rate"]),
        "top20_targeted_customers": int(top20["targeted_customers"]),
        "business_validation_grade": grade,
        "model_use_recommendation": (
            "Use as a ranking model for retention planning. Do not use as the final offer list until "
            "save-worthiness and ROI are added."
        ),
    }

    return pd.DataFrame([row])


def make_model_comparison_readout(model_comparison):
    out = model_comparison.copy()
    out = out.sort_values(["pr_auc", "roc_auc", "top_decile_lift"], ascending=False).reset_index(drop=True)
    out["model_rank"] = np.arange(1, len(out) + 1)

    out["pr_auc_multiple_vs_base"] = np.where(
        out["test_churn_rate"] > 0,
        out["pr_auc"] / out["test_churn_rate"],
        np.nan,
    )

    out["selection_readout"] = np.select(
        [
            out["model_rank"] == 1,
            out["uses_direct_cancel_signals"] == False,
            out["model_name"].str.contains("logistic", case=False, na=False),
        ],
        [
            "Selected model",
            "Sensitivity model without direct cancellation signals",
            "Linear benchmark",
        ],
        default="Challenger model",
    )

    out["business_readout"] = np.select(
        [
            out["top_decile_lift"] >= 5,
            out["top_decile_lift"] >= 3,
            out["top_decile_lift"] >= 1.5,
        ],
        [
            "Excellent lift concentration",
            "Strong lift concentration",
            "Useful lift concentration",
        ],
        default="Weak lift concentration",
    )

    ordered_cols = [
        "model_rank",
        "model_name",
        "model_note",
        "selection_readout",
        "business_readout",
        "uses_direct_cancel_signals",
        "feature_count",
        "test_churn_rate",
        "roc_auc",
        "pr_auc",
        "pr_auc_multiple_vs_base",
        "brier_score",
        "log_loss",
        "top_decile_lift",
        "top_decile_churn_capture_share",
        "top_decile_future_churned_clv_share",
    ]

    return out[[col for col in ordered_cols if col in out.columns]]


def make_decile_business_lift_scorecard(decile):
    out = decile.copy().sort_values("risk_decile")

    out["risk_band_readout"] = np.select(
        [
            out["risk_decile"] == 1,
            out["risk_decile"].between(2, 3),
            out["risk_decile"].between(4, 6),
            out["risk_decile"].between(7, 10),
        ],
        [
            "Critical model validation band",
            "High-risk expansion band",
            "Middle-risk monitoring band",
            "Low-risk holdout/protect band",
        ],
        default="Unassigned",
    )

    out["business_validation_readout"] = np.select(
        [
            out["risk_decile"] == 1,
            out["risk_decile"].between(2, 3),
            out["risk_decile"].between(4, 6),
            out["risk_decile"].between(7, 10),
        ],
        [
            "Should contain the sharpest churn and value concentration.",
            "Useful for campaign expansion after ROI testing.",
            "Watchlist only unless save-worthiness is high.",
            "Avoid paid retention unless customer value is exceptional.",
        ],
        default="Review",
    )

    # A random decile would capture about 10% of churned value.
    # Anything above that shows value concentration from the risk ranking.
    out["decile_value_capture_gap"] = (
        out["future_churned_clv_share"] - 0.10
    )

    return out


def make_threshold_business_policy_scorecard(threshold):
    out = threshold.copy().sort_values("target_population_pct")

    out["marginal_churn_capture"] = out["churn_capture_rate"].diff().fillna(out["churn_capture_rate"])
    out["marginal_future_churned_clv_capture"] = (
        out["future_churned_clv_capture_rate"]
        .diff()
        .fillna(out["future_churned_clv_capture_rate"])
    )

    # This score is only a planning aid. It combines churn capture, value capture,
    # and lift so leadership can compare target-size tradeoffs before ROI scoring.
    out["targeting_efficiency_score"] = (
        0.40 * out["churn_capture_rate"]
        + 0.40 * out["future_churned_clv_capture_rate"]
        + 0.20 * np.minimum(out["lift_vs_portfolio"] / 10, 1)
    )

    # The top 20% risk group is a planning reference, not the final campaign list.
    # Save-worthiness and ROI decide who actually receives paid retention.
    top20_mask = np.isclose(out["target_population_pct"], 0.20)

    out["operating_point_readout"] = np.select(
        [
            out["target_population_pct"] <= 0.05,
            np.isclose(out["target_population_pct"], 0.10),
            top20_mask,
            out["target_population_pct"] > 0.20,
        ],
        [
            "Precision-first executive view",
            "Focused campaign test population",
            "Recommended planning population before save-worthiness scoring",
            "Expansion population; needs ROI guardrails",
        ],
        default="Intermediate planning point",
    )

    out["recommended_for_next_stage"] = np.where(
        top20_mask,
        "Yes - use as planning reference for save-worthiness and ROI simulation",
        "No - keep as sensitivity view",
    )

    return out


def make_calibration_risk_diagnostic(calibration):
    out = calibration.copy().sort_values("calibration_bin")

    # Calibration matters for planning budgets, but the model's primary use here
    # is ranking customers by relative churn risk.
    out["absolute_calibration_gap"] = out["calibration_gap"].abs()

    out["calibration_status"] = np.select(
        [
            out["absolute_calibration_gap"] <= 0.02,
            out["absolute_calibration_gap"] <= 0.05,
            out["absolute_calibration_gap"] > 0.05,
        ],
        [
            "Good",
            "Usable with caution",
            "Needs caution",
        ],
        default="Review",
    )

    out["planning_readout"] = np.select(
        [
            out["calibration_gap"] > 0.05,
            out["calibration_gap"] < -0.05,
        ],
        [
            "Observed churn is higher than predicted; avoid under-budgeting.",
            "Predicted churn is higher than observed; avoid over-discounting.",
        ],
        default="Reasonable planning band",
    )

    return out


def make_risk_tier_value_impact(scored):
    group_cols = [
        "churn_risk_tier",
        "clv_value_tier",
        "retention_budget_tier",
    ]

    out = (
        scored.groupby(group_cols, as_index=False)
        .agg(
            customers=("msno", "count"),
            avg_predicted_churn_probability=("predicted_churn_probability", "mean"),
            actual_future_churn_rate=("actual_future_churn_label", "mean"),
            monthly_value_baseline=("monthly_value_baseline", "sum"),
            monthly_margin_baseline=("monthly_margin_baseline", "sum"),
            profit_adjusted_clv_proxy=("profit_adjusted_clv_proxy", "sum"),
            future_churned_clv_proxy=("future_churned_clv_proxy", "sum"),
            avg_profit_adjusted_clv_proxy=("profit_adjusted_clv_proxy", "mean"),
        )
    )

    portfolio_future_churned_clv = scored["future_churned_clv_proxy"].sum()
    portfolio_customers = len(scored)

    out["customer_share"] = out["customers"] / portfolio_customers
    out["future_churned_clv_share"] = np.where(
        portfolio_future_churned_clv > 0,
        out["future_churned_clv_proxy"] / portfolio_future_churned_clv,
        0,
    )

    # This separates high-risk/high-value customers from high-risk/low-value customers.
    # High churn risk alone is not enough to justify a paid save offer.
    out["risk_value_readout"] = np.select(
        [
            (out["churn_risk_tier"].isin(["Critical risk", "High risk"]))
            & (out["clv_value_tier"].isin(["Elite value", "High value"])),

            (out["churn_risk_tier"].isin(["Critical risk", "High risk"]))
            & (out["clv_value_tier"].isin(["Low value", "No observed value"])),

            (out["churn_risk_tier"].isin(["Low risk"]))
            & (out["clv_value_tier"].isin(["Elite value", "High value"])),
        ],
        [
            "Validate for save-worthiness",
            "High churn risk but weak paid-retention economics",
            "Protect relationship, avoid unnecessary discount",
        ],
        default="Monitor",
    )

    return out.sort_values("future_churned_clv_proxy", ascending=False)


def make_segment_intervention_priority(scored):
    group_cols = [
        "lifecycle_stage",
        "engagement_tier",
        "revenue_tier",
        "clv_value_tier",
        "churn_risk_tier",
        "retention_budget_tier",
        "value_based_action_group",
    ]

    out = (
        scored.groupby(group_cols, as_index=False)
        .agg(
            customers=("msno", "count"),
            avg_predicted_churn_probability=("predicted_churn_probability", "mean"),
            actual_future_churn_rate=("actual_future_churn_label", "mean"),
            monthly_value_baseline=("monthly_value_baseline", "sum"),
            profit_adjusted_clv_proxy=("profit_adjusted_clv_proxy", "sum"),
            future_churned_clv_proxy=("future_churned_clv_proxy", "sum"),
            avg_profit_adjusted_clv_proxy=("profit_adjusted_clv_proxy", "mean"),
        )
    )

    out = out[out["customers"] >= MIN_SEGMENT_SIZE].copy()

    out["business_validation_priority_score"] = (
        out["future_churned_clv_proxy"]
        * (1 + out["avg_predicted_churn_probability"])
        * np.select(
            [
                out["retention_budget_tier"].eq("Premium save budget"),
                out["retention_budget_tier"].eq("Moderate save budget"),
                out["retention_budget_tier"].eq("Low-cost save budget"),
                out["retention_budget_tier"].eq("Automation only"),
                out["retention_budget_tier"].eq("No paid budget"),
            ],
            [1.50, 1.20, 0.80, 0.50, 0.00],
            default=0.75,
        )
    )

    out["validation_recommendation"] = np.select(
        [
            out["churn_risk_tier"].eq("Critical risk")
            & out["clv_value_tier"].isin(["Elite value", "High value"])
            & out["retention_budget_tier"].eq("Premium save budget"),

            out["churn_risk_tier"].isin(["Critical risk", "High risk"])
            & out["retention_budget_tier"].eq("No paid budget"),

            out["churn_risk_tier"].isin(["Critical risk", "High risk"])
            & out["retention_budget_tier"].eq("Automation only"),

            out["churn_risk_tier"].eq("Low risk")
            & out["clv_value_tier"].isin(["Elite value", "High value"]),
        ],
        [
            "Highest priority for save-worthiness layer",
            "Suppress paid offer despite high risk",
            "Use low-cost lifecycle automation",
            "Protect relationship without discount",
        ],
        default="Monitor or test later",
    )

    return out.sort_values("business_validation_priority_score", ascending=False)


def make_cancellation_signal_sensitivity_readout(model_comparison):
    rows = []

    # The no-cancel-signal comparison checks whether the model still ranks churn
    # well without relying only on direct cancellation behavior.
    main = model_comparison[
        model_comparison["model_name"].eq("hist_gradient_boosting")
    ]

    no_cancel = model_comparison[
        model_comparison["model_name"].eq("hist_gradient_boosting_no_cancel_signals")
    ]

    if main.empty or no_cancel.empty:
        return pd.DataFrame(
            [
                {
                    "check_name": "cancel_signal_sensitivity",
                    "status": "NOT_AVAILABLE",
                    "readout": "Main and no-cancel-signal models were not both available.",
                }
            ]
        )

    main = main.iloc[0]
    no_cancel = no_cancel.iloc[0]

    roc_auc_gap = float(main["roc_auc"] - no_cancel["roc_auc"])
    pr_auc_gap = float(main["pr_auc"] - no_cancel["pr_auc"])
    lift_gap = float(main["top_decile_lift"] - no_cancel["top_decile_lift"])

    no_cancel_pr_retention = (
        float(no_cancel["pr_auc"] / main["pr_auc"])
        if float(main["pr_auc"]) > 0
        else np.nan
    )

    if no_cancel_pr_retention >= 0.80 and float(no_cancel["top_decile_lift"]) > 1:
        readout = (
            "The model still ranks churn risk without direct cancellation signals. "
            "Cancellation behavior helps, but broader engagement and transaction patterns also matter."
        )
        status = "PASS"
    else:
        readout = (
            "The model appears highly dependent on direct cancellation signals. "
            "Use it carefully and emphasize this limitation in governance."
        )
        status = "CAUTION"

    rows.append(
        {
            "check_name": "cancel_signal_sensitivity",
            "status": status,
            "main_model_roc_auc": float(main["roc_auc"]),
            "no_cancel_model_roc_auc": float(no_cancel["roc_auc"]),
            "roc_auc_gap": roc_auc_gap,
            "main_model_pr_auc": float(main["pr_auc"]),
            "no_cancel_model_pr_auc": float(no_cancel["pr_auc"]),
            "pr_auc_gap": pr_auc_gap,
            "no_cancel_pr_auc_retention_ratio": no_cancel_pr_retention,
            "main_top_decile_lift": float(main["top_decile_lift"]),
            "no_cancel_top_decile_lift": float(no_cancel["top_decile_lift"]),
            "top_decile_lift_gap": lift_gap,
            "readout": readout,
        }
    )

    return pd.DataFrame(rows)



def make_feature_importance_governance_readout(feature_importance):
    if feature_importance.empty:
        return pd.DataFrame(
            [
                {
                    "feature_rank": 1,
                    "feature": "Not available",
                    "importance_mean": np.nan,
                    "importance_std": np.nan,
                    "signal_family": "Not available",
                    "direct_cancel_importance_share": np.nan,
                    "governance_readout": "Feature importance file was not available.",
                    "portfolio_interpretation": "Review feature importance after rerunning the churn model.",
                }
            ]
        )

    # Grouping features into signal families makes the model easier to govern.
    # The goal is to explain whether risk comes from cancellation, engagement,
    # payment behavior, or customer lifecycle patterns.
    direct_cancel_features = {
        "had_cancel",
        "latest_is_cancel",
        "cancellation_signal_flag",
    }

    revenue_payment_features = {
        "trailing_3mo_revenue",
        "monthly_revenue",
        "monthly_list_price",
        "monthly_discount_amount",
        "avg_paid_per_day",
        "latest_actual_amount_paid",
        "latest_plan_list_price",
        "latest_payment_plan_days",
        "is_auto_renew",
        "latest_is_auto_renew",
        "avg_plan_days",
        "max_plan_days",
        "had_discount",
    }

    engagement_features = {
        "active_days",
        "total_secs",
        "total_plays",
        "num_unq",
        "completion_rate_proxy",
        "engagement_score",
        "trailing_3mo_engagement_score",
        "trailing_3mo_active_days",
        "engagement_change_pct",
        "activity_change_pct",
        "revenue_change_pct",
        "no_recent_activity_flag",
        "major_activity_drop_flag",
    }

    customer_profile_features = {
        "age",
        "tenure_months",
        "city",
        "registered_via",
        "gender",
        "lifecycle_stage",
        "engagement_tier",
        "revenue_tier",
    }

    def classify_feature(feature):
        feature = str(feature)

        if feature in direct_cancel_features:
            return "Direct cancellation signal"

        if feature in revenue_payment_features:
            return "Revenue and payment behavior"

        if feature in engagement_features:
            return "Engagement behavior"

        if feature in customer_profile_features:
            return "Customer profile and lifecycle"

        return "Other modeled signal"

    fi = feature_importance.copy()
    fi["importance_mean"] = pd.to_numeric(fi["importance_mean"], errors="coerce").fillna(0)
    fi["importance_std"] = pd.to_numeric(fi["importance_std"], errors="coerce").fillna(0)

    fi["positive_importance"] = fi["importance_mean"].clip(lower=0)
    total_positive_importance = fi["positive_importance"].sum()

    direct_cancel_importance = fi.loc[
        fi["feature"].isin(direct_cancel_features),
        "positive_importance",
    ].sum()

    # This share is a governance diagnostic, not a model-selection rule by itself.
    # A high cancellation-signal share means the model may be less useful for early intervention.
    direct_cancel_share = (
        direct_cancel_importance / total_positive_importance
        if total_positive_importance > 0
        else np.nan
    )

    top = fi.sort_values("importance_mean", ascending=False).head(20).copy()
    top["feature_rank"] = np.arange(1, len(top) + 1)
    top["signal_family"] = top["feature"].apply(classify_feature)
    top["direct_cancel_importance_share"] = direct_cancel_share

    if pd.isna(direct_cancel_share):
        governance_readout = "Feature importance could not be summarized."
    elif direct_cancel_share >= 0.70:
        governance_readout = (
            "Direct cancellation signals dominate feature importance. Use the model carefully "
            "and emphasize cancellation-signal dependence in governance."
        )
    elif direct_cancel_share >= 0.35:
        governance_readout = (
            "Cancellation signals are important, but the model also uses broader revenue, "
            "payment, engagement, and lifecycle behavior."
        )
    else:
        governance_readout = (
            "Feature importance is not dominated by cancellation signals. The model appears "
            "to learn broader customer behavior patterns."
        )

    top["governance_readout"] = governance_readout

    top["portfolio_interpretation"] = np.select(
        [
            top["signal_family"].eq("Direct cancellation signal"),
            top["signal_family"].eq("Revenue and payment behavior"),
            top["signal_family"].eq("Engagement behavior"),
            top["signal_family"].eq("Customer profile and lifecycle"),
        ],
        [
            "Strong near-term churn signal; useful but should not be the only basis for retention spend.",
            "Commercial behavior signal; useful for identifying value-linked churn risk.",
            "Usage behavior signal; useful for lifecycle intervention timing.",
            "Segmentation signal; useful for explaining where risk concentrates.",
        ],
        default="Supporting model signal.",
    )

    ordered_cols = [
        "feature_rank",
        "feature",
        "importance_mean",
        "importance_std",
        "signal_family",
        "direct_cancel_importance_share",
        "governance_readout",
        "portfolio_interpretation",
    ]

    return top[ordered_cols]


def make_leakage_and_governance_check(feature_audit, training_manifest):
    rows = []

    # This reuses the churn-model audit file to make sure future-value fields,
    # target fields, and final action labels stayed out of the feature matrix.
    forbidden = set(feature_audit.get("forbidden_model_features", []))
    numeric = set(feature_audit.get("numeric_features_main_model", []))
    categorical = set(feature_audit.get("categorical_features", []))
    used = numeric.union(categorical)

    forbidden_used = sorted(forbidden.intersection(used))

    rows.append(
        {
            "governance_check": "forbidden_features_not_used",
            "status": "PASS" if len(forbidden_used) == 0 else "FAIL",
            "value": ", ".join(forbidden_used) if forbidden_used else "None",
            "readout": "Target, future-churned value, and final action fields should not be model predictors.",
        }
    )

    excluded_business_fields = set(
        training_manifest.get("excluded_from_features_but_used_for_business_evaluation", [])
    )

    rows.append(
        {
            "governance_check": "business_value_used_after_scoring",
            "status": "PASS" if "future_churned_clv_proxy" in excluded_business_fields else "REVIEW",
            "value": ", ".join(sorted(excluded_business_fields)) if excluded_business_fields else "Not found",
            "readout": "CLV and future-churned value should be used for validation after scoring, not as model inputs.",
        }
    )

    rows.append(
        {
            "governance_check": "sensitivity_model_trained",
            "status": "PASS" if "sensitivity" in str(training_manifest).lower() else "REVIEW",
            "value": training_manifest.get("sensitivity_test", "Not found"),
            "readout": "A no-cancellation-signal challenger helps test whether the model is too dependent on obvious cancel flags.",
        }
    )

    rows.append(
        {
            "governance_check": "model_output_not_final_targeting",
            "status": "PASS",
            "value": "Churn risk only",
            "readout": "The model ranks churn risk. Save-worthiness and ROI decide final retention actions.",
        }
    )

    return pd.DataFrame(rows)


def choose_recommended_operating_point(threshold_scorecard):
    # Choose a planning threshold that captures strong churned value without
    # expanding too far before economic targeting is applied.
    candidates = threshold_scorecard[
        threshold_scorecard["target_population_pct"] <= 0.20
    ].copy()

    strong_value = candidates[
        candidates["future_churned_clv_capture_rate"] >= 0.75
    ]

    if not strong_value.empty:
        selected = strong_value.sort_values("target_population_pct").iloc[0]
    else:
        selected = threshold_scorecard[
            np.isclose(threshold_scorecard["target_population_pct"], 0.20)
        ].iloc[0]

    return selected


def write_sql_reference():
    sql_path = SQL_DIR / "07_model_validation_business_impact.sql"

    lines = [
        "-- 07_model_validation_business_impact.sql",
        "-- SQL reference queries for model validation outputs.",
        "-- The Python script creates the final CSVs, but these queries document the business logic.",
        "",
        "-- Top risk decile lift and value capture",
        "SELECT",
        "    risk_decile,",
        "    customers,",
        "    observed_churn_rate,",
        "    lift_vs_portfolio,",
        "    churn_capture_share,",
        "    cumulative_churn_capture_share,",
        "    future_churned_clv_share,",
        "    cumulative_future_churned_clv_share",
        "FROM read_csv_auto('data/processed/churn_model_outputs/01_risk_decile_business_summary.csv')",
        "ORDER BY risk_decile;",
        "",
        "-- Operating threshold policy simulation",
        "SELECT",
        "    target_population_pct,",
        "    targeted_customers,",
        "    score_threshold,",
        "    observed_churn_rate_targeted,",
        "    lift_vs_portfolio,",
        "    churn_capture_rate,",
        "    future_churned_clv_capture_rate",
        "FROM read_csv_auto('data/processed/churn_model_outputs/02_targeting_threshold_simulation.csv')",
        "ORDER BY target_population_pct;",
        "",
        "-- Calibration diagnostic",
        "SELECT",
        "    calibration_bin,",
        "    customers,",
        "    avg_predicted_churn_probability,",
        "    observed_churn_rate,",
        "    calibration_gap",
        "FROM read_csv_auto('data/processed/churn_model_outputs/03_calibration_by_score_bin.csv')",
        "ORDER BY calibration_bin;",
        "",
        "-- Risk tier and value concentration after scoring",
        "SELECT",
        "    churn_risk_tier,",
        "    clv_value_tier,",
        "    retention_budget_tier,",
        "    customers,",
        "    avg_predicted_churn_probability,",
        "    actual_future_churn_rate,",
        "    profit_adjusted_clv_proxy,",
        "    future_churned_clv_proxy",
        "FROM read_csv_auto('data/processed/model_validation_outputs/05_risk_tier_value_impact.csv')",
        "ORDER BY future_churned_clv_proxy DESC;",
    ]

    with open(sql_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved SQL reference file: {sql_path}")


def validate_outputs(outputs):
    checks = []

    def add_check(check_name, value, passed, notes):
        checks.append(
            {
                "check_name": check_name,
                "status": "PASS" if passed else "FAIL",
                "value": value,
                "notes": notes,
            }
        )

    kpi = outputs["00_validation_kpi_summary"].iloc[0]
    decile = outputs["02_decile_business_lift_scorecard"]
    threshold = outputs["03_threshold_business_policy_scorecard"]
    calibration = outputs["04_calibration_risk_diagnostic"]
    governance = outputs["08_leakage_and_governance_check"]
    feature_importance_governance = outputs["09_feature_importance_governance_readout"]

    top_decile = decile[decile["risk_decile"] == 1].iloc[0]

    add_check(
        "best_model_exists",
        kpi["best_model"],
        isinstance(kpi["best_model"], str) and len(kpi["best_model"]) > 0,
        "Best model should be identified.",
    )

    add_check(
        "roc_auc_above_random",
        round(float(kpi["roc_auc"]), 6),
        float(kpi["roc_auc"]) > 0.50,
        "ROC-AUC should beat random ranking.",
    )

    add_check(
        "pr_auc_above_base_rate",
        round(float(kpi["pr_auc_multiple_vs_base_churn"]), 6),
        float(kpi["pr_auc_multiple_vs_base_churn"]) > 1.0,
        "PR-AUC should beat the base churn rate.",
    )

    add_check(
        "top_decile_lift_above_one",
        round(float(top_decile["lift_vs_portfolio"]), 6),
        float(top_decile["lift_vs_portfolio"]) > 1.0,
        "Top decile should concentrate churn risk.",
    )

    add_check(
        "top_decile_captures_more_than_population_share",
        round(float(top_decile["churn_capture_share"]), 6),
        float(top_decile["churn_capture_share"]) > 0.10,
        "Top decile should capture more than 10% of churners.",
    )

    add_check(
        "threshold_table_has_top20",
        bool(np.isclose(threshold["target_population_pct"], 0.20).any()),
        bool(np.isclose(threshold["target_population_pct"], 0.20).any()),
        "Threshold table should include top 20% planning point.",
    )

    avg_abs_calibration_gap = float(calibration["absolute_calibration_gap"].mean())

    add_check(
        "calibration_table_created",
        len(calibration),
        len(calibration) >= 5,
        "Calibration diagnostic should have multiple bins.",
    )

    add_check(
        "average_abs_calibration_gap_reasonable",
        round(avg_abs_calibration_gap, 6),
        avg_abs_calibration_gap <= 0.10,
        "Average calibration gap should be reasonable for planning.",
    )

    governance_failed = governance[governance["status"] == "FAIL"]

    add_check(
        "governance_checks_no_failures",
        len(governance_failed),
        len(governance_failed) == 0,
        "Governance checks should not contain failures.",
    )

    add_check(
        "feature_importance_governance_created",
        len(feature_importance_governance),
        len(feature_importance_governance) > 0,
        "Feature-importance governance output should exist.",
    )

    report = pd.DataFrame(checks)
    report.to_csv(OUTPUT_DIR / "_model_validation_business_impact_validation_report.csv", index=False)

    with open(OUTPUT_DIR / "_model_validation_business_impact_validation_report.json", "w") as f:
        json.dump(clean_for_json(checks), f, indent=2)

    print("\nValidation report:")
    print(report.to_string(index=False))

    failed = report[report["status"] == "FAIL"]
    if len(failed) > 0:
        raise ValueError("Model validation business-impact checks failed. Review outputs before moving forward.")

    return report


def write_executive_summary(outputs):
    kpi = outputs["00_validation_kpi_summary"].iloc[0]
    threshold = outputs["03_threshold_business_policy_scorecard"]
    calibration = outputs["04_calibration_risk_diagnostic"]
    risk_value = outputs["05_risk_tier_value_impact"]
    segment_priority = outputs["06_segment_intervention_priority"]
    sensitivity = outputs["07_cancellation_signal_sensitivity_readout"]
    governance = outputs["08_leakage_and_governance_check"]
    feature_importance_governance = outputs["09_feature_importance_governance_readout"]

    operating = choose_recommended_operating_point(threshold)
    top_segment = segment_priority.iloc[0]
    top_risk_value = risk_value.iloc[0]
    top_feature = feature_importance_governance.iloc[0]

    avg_abs_calibration_gap = float(calibration["absolute_calibration_gap"].mean())
    max_abs_calibration_gap = float(calibration["absolute_calibration_gap"].max())

    # Calibration affects how confidently the probabilities can be used for planning.
    # Even with imperfect calibration, a strong ranking model can still be useful
    # for prioritization if lift and value capture are strong.
    if avg_abs_calibration_gap <= 0.02:
        calibration_readout = "Calibration is strong enough for planning."
    elif avg_abs_calibration_gap <= 0.05:
        calibration_readout = "Calibration is usable, but threshold decisions should still be tested."
    else:
        calibration_readout = "Use the model more as a ranking engine than as a perfectly calibrated probability engine."

    sensitivity_readout = (
        sensitivity.iloc[0]["readout"]
        if "readout" in sensitivity.columns
        else "Sensitivity readout not available."
    )

    governance_status = (
        "PASS"
        if (governance["status"] != "FAIL").all()
        else "FAIL"
    )

    summary = {
        "best_model": kpi["best_model"],
        "business_validation_grade": kpi["business_validation_grade"],
        "roc_auc": float(kpi["roc_auc"]),
        "pr_auc": float(kpi["pr_auc"]),
        "pr_auc_multiple_vs_base_churn": float(kpi["pr_auc_multiple_vs_base_churn"]),
        "brier_score": float(kpi["brier_score"]),
        "top_decile_lift": float(kpi["top_decile_lift"]),
        "top_decile_churn_capture_share": float(kpi["top_decile_churn_capture_share"]),
        "top_decile_future_churned_clv_share": float(kpi["top_decile_future_churned_clv_share"]),
        "recommended_operating_point": {
            "target_population_pct": float(operating["target_population_pct"]),
            "targeted_customers": int(operating["targeted_customers"]),
            "score_threshold": float(operating["score_threshold"]),
            "churn_capture_rate": float(operating["churn_capture_rate"]),
            "future_churned_clv_capture_rate": float(operating["future_churned_clv_capture_rate"]),
        },
        "calibration": {
            "avg_abs_calibration_gap": avg_abs_calibration_gap,
            "max_abs_calibration_gap": max_abs_calibration_gap,
            "readout": calibration_readout,
        },
        "top_risk_value_group": {
            "churn_risk_tier": top_risk_value["churn_risk_tier"],
            "clv_value_tier": top_risk_value["clv_value_tier"],
            "retention_budget_tier": top_risk_value["retention_budget_tier"],
            "customers": int(top_risk_value["customers"]),
            "future_churned_clv_proxy": float(top_risk_value["future_churned_clv_proxy"]),
        },
        "top_segment_priority": {
            "lifecycle_stage": top_segment["lifecycle_stage"],
            "engagement_tier": top_segment["engagement_tier"],
            "revenue_tier": top_segment["revenue_tier"],
            "clv_value_tier": top_segment["clv_value_tier"],
            "churn_risk_tier": top_segment["churn_risk_tier"],
            "retention_budget_tier": top_segment["retention_budget_tier"],
            "customers": int(top_segment["customers"]),
            "future_churned_clv_proxy": float(top_segment["future_churned_clv_proxy"]),
            "validation_recommendation": top_segment["validation_recommendation"],
        },
        "sensitivity_readout": str(sensitivity_readout),
        "feature_importance_governance": {
            "top_feature": str(top_feature["feature"]),
            "top_feature_family": str(top_feature["signal_family"]),
            "importance_mean": float(top_feature["importance_mean"]),
            "direct_cancel_importance_share": (
                None
                if pd.isna(top_feature["direct_cancel_importance_share"])
                else float(top_feature["direct_cancel_importance_share"])
            ),
            "governance_readout": str(top_feature["governance_readout"]),
        },
        "governance_status": governance_status,
    }

    with open(OUTPUT_DIR / "_model_validation_business_impact_summary.json", "w") as f:
        json.dump(clean_for_json(summary), f, indent=2)

    lines = [
        "# Model Validation and Business Impact Summary",
        "",
        f"Best model: {summary['best_model']}",
        f"Business validation grade: {summary['business_validation_grade']}",
        f"ROC-AUC: {summary['roc_auc']:.4f}",
        f"PR-AUC: {summary['pr_auc']:.4f}",
        f"PR-AUC multiple vs base churn rate: {summary['pr_auc_multiple_vs_base_churn']:.2f}x",
        f"Brier score: {summary['brier_score']:.4f}",
        "",
        "Business lift validation:",
        f"- Top risk decile lift: {summary['top_decile_lift']:.2f}x",
        f"- Top risk decile captures {summary['top_decile_churn_capture_share']:.2%} of churners.",
        f"- Top risk decile captures {summary['top_decile_future_churned_clv_share']:.2%} of future churned CLV proxy.",
        "",
        "Recommended planning threshold:",
        (
            f"- Use the top {summary['recommended_operating_point']['target_population_pct']:.0%} risk population "
            f"as the planning reference before save-worthiness scoring."
        ),
        f"- Targeted customers at that point: {summary['recommended_operating_point']['targeted_customers']:,}",
        f"- Churn capture rate: {summary['recommended_operating_point']['churn_capture_rate']:.2%}",
        f"- Future churned CLV capture rate: {summary['recommended_operating_point']['future_churned_clv_capture_rate']:.2%}",
        "",
        "Calibration readout:",
        f"- Average absolute calibration gap: {summary['calibration']['avg_abs_calibration_gap']:.2%}",
        f"- Maximum absolute calibration gap: {summary['calibration']['max_abs_calibration_gap']:.2%}",
        f"- {summary['calibration']['readout']}",
        "",
        "Highest business-impact validation group:",
        (
            f"- {summary['top_risk_value_group']['churn_risk_tier']} / "
            f"{summary['top_risk_value_group']['clv_value_tier']} / "
            f"{summary['top_risk_value_group']['retention_budget_tier']} contains "
            f"{summary['top_risk_value_group']['customers']:,} customers and "
            f"{summary['top_risk_value_group']['future_churned_clv_proxy']:,.0f} future churned CLV proxy."
        ),
        "",
        "Highest-priority segment for the next layer:",
        (
            f"- {summary['top_segment_priority']['lifecycle_stage']} / "
            f"{summary['top_segment_priority']['engagement_tier']} / "
            f"{summary['top_segment_priority']['revenue_tier']} / "
            f"{summary['top_segment_priority']['clv_value_tier']} / "
            f"{summary['top_segment_priority']['churn_risk_tier']} / "
            f"{summary['top_segment_priority']['retention_budget_tier']}"
        ),
        f"- Recommendation: {summary['top_segment_priority']['validation_recommendation']}",
        "",
        "Cancellation-signal sensitivity:",
        f"- {summary['sensitivity_readout']}",
        "",
        "Feature-importance governance:",
        (
            f"- Top permutation-importance driver: "
            f"{summary['feature_importance_governance']['top_feature']} "
            f"({summary['feature_importance_governance']['top_feature_family']})."
        ),
        (
            f"- Direct cancellation-signal importance share: "
            f"{summary['feature_importance_governance']['direct_cancel_importance_share']:.2%}"
            if summary['feature_importance_governance']['direct_cancel_importance_share'] is not None
            else "- Direct cancellation-signal importance share: not available."
        ),
        f"- {summary['feature_importance_governance']['governance_readout']}",
        "",
        "Governance readout:",
        f"- Governance status: {summary['governance_status']}",
        (
            "- The model is validated as a churn-risk ranking engine. It is not the final retention action list. "
            "The next script should combine predicted churn probability with CLV, margin, intervention cost, "
            "expected response, and lifecycle context."
        ),
    ]

    summary_path = OUTPUT_DIR / "_executive_model_validation_business_impact_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved {summary_path}")
    print(f"Saved {OUTPUT_DIR / '_model_validation_business_impact_summary.json'}")


def main():
    print("\nRunning model validation and business impact analysis...")

    inputs = load_inputs()

    kpi_summary = make_validation_kpi_summary(inputs)
    model_comparison_readout = make_model_comparison_readout(inputs["model_comparison"])
    decile_scorecard = make_decile_business_lift_scorecard(inputs["decile"])
    threshold_scorecard = make_threshold_business_policy_scorecard(inputs["threshold"])
    calibration_diagnostic = make_calibration_risk_diagnostic(inputs["calibration"])
    risk_tier_value_impact = make_risk_tier_value_impact(inputs["scored"])
    segment_priority = make_segment_intervention_priority(inputs["scored"])
    sensitivity = make_cancellation_signal_sensitivity_readout(inputs["model_comparison"])
    governance = make_leakage_and_governance_check(
        feature_audit=inputs["feature_audit"],
        training_manifest=inputs["training_manifest"],
    )
    feature_importance_governance = make_feature_importance_governance_readout(
        inputs["feature_importance"]
    )

    outputs = {
        "00_validation_kpi_summary": kpi_summary,
        "01_model_comparison_readout": model_comparison_readout,
        "02_decile_business_lift_scorecard": decile_scorecard,
        "03_threshold_business_policy_scorecard": threshold_scorecard,
        "04_calibration_risk_diagnostic": calibration_diagnostic,
        "05_risk_tier_value_impact": risk_tier_value_impact,
        "06_segment_intervention_priority": segment_priority,
        "07_cancellation_signal_sensitivity_readout": sensitivity,
        "08_leakage_and_governance_check": governance,
        "09_feature_importance_governance_readout": feature_importance_governance,
    }

    write_sql_reference()

    for name, df in outputs.items():
        output_path = OUTPUT_DIR / f"{name}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path} | rows={len(df):,} cols={df.shape[1]:,}")

    validation_report = validate_outputs(outputs)
    write_executive_summary(outputs)

    print("\n07_model_validation_business_impact.py complete.")
    

if __name__ == "__main__":
    main()
