"""
08_save_worthiness_scoring.py

Save-worthiness scoring layer for the Customer Revenue Recovery & Retention ROI Engine.

This is the first true decision-engine script in the project.

The churn model answers:
Who is likely to leave?

The CLV layer answers:
Who is valuable?

The lifecycle layer answers:
Where does timing and lifecycle context matter?

This script combines those into the question leadership actually cares about:

Who is worth saving?

I intentionally do not target customers just because they have high churn risk.
A customer can be high risk and still not deserve a paid offer if the value,
response expectation, or economics do not justify it.

Core formula:
expected_saved_clv_proxy =
    predicted_churn_probability
    * profit_adjusted_clv_proxy
    * expected_intervention_response_rate

net_save_value_proxy =
    expected_saved_clv_proxy - intervention_cost_proxy

Important:
profit_adjusted_clv_proxy already reflects margin assumptions from the CLV layer.
So I do not multiply by margin again here.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed")
CHURN_DIR = PROCESSED_DIR / "churn_model_outputs"
SURVIVAL_DIR = PROCESSED_DIR / "survival_lifecycle_outputs"
VALIDATION_DIR = PROCESSED_DIR / "model_validation_outputs"
OUTPUT_DIR = PROCESSED_DIR / "save_worthiness_outputs"
SQL_DIR = Path("sql")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SQL_DIR.mkdir(parents=True, exist_ok=True)

CHURN_SCORED_PATH = CHURN_DIR / "churn_scored_customers.parquet"
LIFECYCLE_FEATURES_PATH = SURVIVAL_DIR / "customer_lifecycle_survival_features.parquet"

MIN_SEGMENT_SIZE = 250

# These are planning assumptions, not fitted model outputs.
# I keep them explicit so the ROI simulator can later stress-test response rates,
# intervention costs, and campaign economics under different scenarios.
ASSUMPTIONS = {
    "economic_note": (
        "profit_adjusted_clv_proxy already includes the gross-margin assumption from the CLV layer. "
        "This script does not multiply by margin again."
    ),
    "response_rate_assumptions": {
        "Immediate premium save offer": 0.14,
        "Premium win-back review": 0.12,
        "High-value targeted retention": 0.08,
        "Targeted lifecycle nurture": 0.05,
        "Low-cost automated retention": 0.025,
        "Protect without discount": 0.01,
        "Monitor only": 0.00,
        "Suppress paid offer": 0.00,
    },
    "intervention_cost_assumptions": {
        "Immediate premium save offer": 18.00,
        "Premium win-back review": 12.00,
        "High-value targeted retention": 7.50,
        "Targeted lifecycle nurture": 3.00,
        "Low-cost automated retention": 0.75,
        "Protect without discount": 0.25,
        "Monitor only": 0.00,
        "Suppress paid offer": 0.00,
    },
    "scoring_note": (
        "Future churn labels and future_churned_clv_proxy are used only for retrospective validation. "
        "They are not used to calculate save-worthiness."
    ),
}


REQUIRED_CHURN_COLUMNS = [
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


OPTIONAL_LIFECYCLE_COLUMNS = [
    "msno",
    "lifecycle_duration_months",
    "observed_active_months",
    "observed_month_coverage_rate",
    "value_risk_quadrant",
    "lifecycle_strategy_readout",
]


def require_file(path: Path, upstream_script: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run {upstream_script} first.")


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if pd.isna(obj):
        return None
    return obj


def load_inputs() -> pd.DataFrame:
    require_file(CHURN_SCORED_PATH, "src/05_churn_model.py")
    require_file(LIFECYCLE_FEATURES_PATH, "src/06_survival_lifecycle_analysis.py")

    print("\nLoading churn scores and lifecycle features...")

    churn = pd.read_parquet(CHURN_SCORED_PATH)
    missing = sorted(set(REQUIRED_CHURN_COLUMNS) - set(churn.columns))
    if missing:
        raise ValueError(f"Missing required columns from churn scored file: {missing}")

    lifecycle = pd.read_parquet(LIFECYCLE_FEATURES_PATH)

    available_lifecycle_cols = [
        col for col in OPTIONAL_LIFECYCLE_COLUMNS
        if col in lifecycle.columns
    ]

    lifecycle = lifecycle[available_lifecycle_cols].drop_duplicates("msno")

    df = churn.merge(
        lifecycle,
        on="msno",
        how="left",
        suffixes=("", "_lifecycle"),
    )

    for col in [
        "predicted_churn_probability",
        "monthly_value_baseline",
        "monthly_margin_baseline",
        "annual_revenue_run_rate_proxy",
        "annual_margin_run_rate_proxy",
        "profit_adjusted_clv_proxy",
        "future_churned_clv_proxy",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # The actual future churn label is carried forward for validation and readouts only.
    # It is not used to calculate save-worthiness, because that would leak the outcome
    # into the targeting decision.
    df["actual_future_churn_label"] = pd.to_numeric(
        df["actual_future_churn_label"],
        errors="coerce",
    ).fillna(0).astype(int)

    df["predicted_churn_probability"] = df["predicted_churn_probability"].clip(0, 1)
    df["profit_adjusted_clv_proxy"] = df["profit_adjusted_clv_proxy"].clip(lower=0)

    if "value_risk_quadrant" not in df.columns:
        df["value_risk_quadrant"] = "Unknown value-risk quadrant"

    if "lifecycle_strategy_readout" not in df.columns:
        df["lifecycle_strategy_readout"] = "Lifecycle strategy not available"

    print(f"Loaded scored customers: {len(df):,}")
    print(f"Average predicted churn probability: {df['predicted_churn_probability'].mean():.2%}")

    return df


def recommend_intervention_action(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    high_value = out["clv_value_tier"].isin(["Elite value", "High value"])
    medium_or_high_value = out["clv_value_tier"].isin(["Elite value", "High value", "Core value"])
    high_risk = out["churn_risk_tier"].isin(["Critical risk", "High risk"])
    critical_risk = out["churn_risk_tier"].eq("Critical risk")
    premium_budget = out["retention_budget_tier"].eq("Premium save budget")
    moderate_budget = out["retention_budget_tier"].eq("Moderate save budget")
    automation_budget = out["retention_budget_tier"].eq("Automation only")
    no_paid_budget = out["retention_budget_tier"].eq("No paid budget")
    low_risk_high_value = out["value_risk_quadrant"].eq("Low risk / high value")
    no_value = out["profit_adjusted_clv_proxy"].le(0)

        # Rule order matters here. I suppress customers with no economic case first,
        # then assign the highest-touch actions to customers where churn risk, CLV,
        # and budget tier justify a paid intervention.  
    conditions = [
        no_value | no_paid_budget,
        critical_risk & high_value & premium_budget,
        high_risk & high_value & out["value_based_action_group"].eq("Premium win-back"),
        high_risk & high_value & premium_budget,
        high_risk & medium_or_high_value & moderate_budget,
        high_risk & automation_budget,
        low_risk_high_value,
    ]

    choices = [
        "Suppress paid offer",
        "Immediate premium save offer",
        "Premium win-back review",
        "High-value targeted retention",
        "Targeted lifecycle nurture",
        "Low-cost automated retention",
        "Protect without discount",
    ]

    out["recommended_retention_action"] = np.select(
        conditions,
        choices,
        default="Monitor only",
    )

    return out


def attach_assumptions_and_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    response_map = ASSUMPTIONS["response_rate_assumptions"]
    cost_map = ASSUMPTIONS["intervention_cost_assumptions"]

    out["expected_intervention_response_rate"] = (
        out["recommended_retention_action"]
        .map(response_map)
        .fillna(0)
        .astype(float)
    )

    out["intervention_cost_proxy"] = (
        out["recommended_retention_action"]
        .map(cost_map)
        .fillna(0)
        .astype(float)
    )

    # Decision-time value at risk uses only predicted churn probability and CLV.
    # Future churned CLV is kept for retrospective validation, not for scoring.
    out["gross_value_at_risk_proxy"] = (
        out["predicted_churn_probability"]
        * out["profit_adjusted_clv_proxy"]
    )

    out["expected_saved_clv_proxy"] = (
        out["gross_value_at_risk_proxy"]
        * out["expected_intervention_response_rate"]
    )

    out["net_save_value_proxy"] = (
        out["expected_saved_clv_proxy"]
        - out["intervention_cost_proxy"]
    )

    denominator = out["gross_value_at_risk_proxy"].replace(0, np.nan)
    out["break_even_response_rate"] = out["intervention_cost_proxy"] / denominator

    # This is a net ROI proxy: expected net save value per dollar of intervention cost.
    # Gross return would use expected_saved_clv_proxy / intervention_cost_proxy instead.
    out["expected_roi_proxy"] = np.where(
        out["intervention_cost_proxy"] > 0,
        out["net_save_value_proxy"] / out["intervention_cost_proxy"],
        np.nan,
    )

    out["paid_offer_economically_positive"] = (
        out["intervention_cost_proxy"].gt(0)
        & out["net_save_value_proxy"].gt(0)
    )

    out["should_receive_paid_offer"] = (
        out["paid_offer_economically_positive"]
        & ~out["recommended_retention_action"].isin(["Suppress paid offer", "Monitor only"])
    )

    out["suppression_reason"] = np.select(
        [
            out["recommended_retention_action"].eq("Suppress paid offer")
            & out["profit_adjusted_clv_proxy"].le(0),

            out["recommended_retention_action"].eq("Suppress paid offer")
            & out["retention_budget_tier"].eq("No paid budget"),

            out["intervention_cost_proxy"].gt(0)
            & out["net_save_value_proxy"].le(0),

            out["recommended_retention_action"].eq("Monitor only"),
        ],
        [
            "No positive CLV proxy",
            "No paid budget tier",
            "Expected save value does not clear intervention cost",
            "Monitor only before ROI testing",
        ],
        default="Not suppressed",
    )
    # Only economically positive paid-offer candidates are ranked.
    # Non-economic customers should not be forced into a priority tier just because
    # they have high churn probability.
    eligible = out["net_save_value_proxy"].gt(0) & out["should_receive_paid_offer"]

    out["save_worthiness_rank"] = np.nan
    out.loc[eligible, "save_worthiness_rank"] = (
        out.loc[eligible, "net_save_value_proxy"]
        .rank(method="first", ascending=False)
    )

    eligible_count = int(eligible.sum())

    if eligible_count > 1:
        out.loc[eligible, "save_worthiness_score"] = (
            100
            * (
                1
                - (
                    out.loc[eligible, "save_worthiness_rank"] - 1
                )
                / (eligible_count - 1)
            )
        )
    elif eligible_count == 1:
        out.loc[eligible, "save_worthiness_score"] = 100
    else:
        out["save_worthiness_score"] = 0

    out["save_worthiness_score"] = out["save_worthiness_score"].fillna(0)

    out["save_priority_tier"] = "Not targeted"

    if eligible_count > 0:
        percentile_rank = out.loc[eligible, "save_worthiness_rank"] / eligible_count

        tier_values = np.select(
            [
                percentile_rank.le(0.01),
                percentile_rank.le(0.05),
                percentile_rank.le(0.10),
                percentile_rank.le(0.20),
            ],
            [
                "Tier 1 - Executive save priority",
                "Tier 2 - Premium save priority",
                "Tier 3 - Targeted save priority",
                "Tier 4 - Test-and-learn save pool",
            ],
            default="Positive ROI monitor pool",
        )

        out.loc[eligible, "save_priority_tier"] = tier_values

    out.loc[
        out["recommended_retention_action"].eq("Suppress paid offer"),
        "save_priority_tier",
    ] = "Suppressed from paid retention"

    out.loc[
        out["recommended_retention_action"].eq("Protect without discount"),
        "save_priority_tier",
    ] = "Relationship protection only"

    out["final_targeting_readout"] = np.select(
        [
            out["save_priority_tier"].eq("Tier 1 - Executive save priority"),
            out["save_priority_tier"].eq("Tier 2 - Premium save priority"),
            out["save_priority_tier"].eq("Tier 3 - Targeted save priority"),
            out["save_priority_tier"].eq("Tier 4 - Test-and-learn save pool"),
            out["save_priority_tier"].eq("Suppressed from paid retention"),
            out["save_priority_tier"].eq("Relationship protection only"),
        ],
        [
            "Highest expected economic save opportunity",
            "Strong paid-retention candidate",
            "Targeted paid-retention candidate",
            "Use for controlled campaign testing",
            "Do not spend paid retention budget",
            "Protect experience, avoid discounting",
        ],
        default="Monitor until ROI simulation",
    )

    return out.sort_values(
        ["should_receive_paid_offer", "net_save_value_proxy", "predicted_churn_probability"],
        ascending=[False, False, False],
    )


def make_portfolio_summary(scored: pd.DataFrame) -> pd.DataFrame:
    paid = scored["should_receive_paid_offer"]

    row = {
        "customers": len(scored),
        "avg_predicted_churn_probability": scored["predicted_churn_probability"].mean(),
        "actual_future_churn_rate": scored["actual_future_churn_label"].mean(),
        "profit_adjusted_clv_proxy": scored["profit_adjusted_clv_proxy"].sum(),
        "future_churned_clv_proxy": scored["future_churned_clv_proxy"].sum(),
        "gross_value_at_risk_proxy": scored["gross_value_at_risk_proxy"].sum(),
        "expected_saved_clv_proxy": scored["expected_saved_clv_proxy"].sum(),
        "intervention_cost_proxy": scored["intervention_cost_proxy"].sum(),
        "net_save_value_proxy": scored["net_save_value_proxy"].sum(),
        "paid_offer_customers": int(paid.sum()),
        "paid_offer_rate": paid.mean(),
        "paid_offer_expected_saved_clv_proxy": scored.loc[paid, "expected_saved_clv_proxy"].sum(),
        "paid_offer_intervention_cost_proxy": scored.loc[paid, "intervention_cost_proxy"].sum(),
        "paid_offer_net_save_value_proxy": scored.loc[paid, "net_save_value_proxy"].sum(),
        "paid_offer_actual_churn_rate": scored.loc[paid, "actual_future_churn_label"].mean() if paid.any() else 0,
        "paid_offer_future_churned_clv_proxy": scored.loc[paid, "future_churned_clv_proxy"].sum(),
    }

    row["paid_offer_roi_proxy"] = (
        row["paid_offer_net_save_value_proxy"] / row["paid_offer_intervention_cost_proxy"]
        if row["paid_offer_intervention_cost_proxy"] > 0
        else np.nan
    )

    return pd.DataFrame([row])


def make_action_summary(scored: pd.DataFrame) -> pd.DataFrame:
    out = (
        scored.groupby("recommended_retention_action", as_index=False)
        .agg(
            customers=("msno", "count"),
            avg_predicted_churn_probability=("predicted_churn_probability", "mean"),
            actual_future_churn_rate=("actual_future_churn_label", "mean"),
            profit_adjusted_clv_proxy=("profit_adjusted_clv_proxy", "sum"),
            future_churned_clv_proxy=("future_churned_clv_proxy", "sum"),
            gross_value_at_risk_proxy=("gross_value_at_risk_proxy", "sum"),
            expected_saved_clv_proxy=("expected_saved_clv_proxy", "sum"),
            intervention_cost_proxy=("intervention_cost_proxy", "sum"),
            net_save_value_proxy=("net_save_value_proxy", "sum"),
            avg_expected_roi_proxy=("expected_roi_proxy", "mean"),
            paid_offer_customers=("should_receive_paid_offer", "sum"),
        )
    )

    out["roi_proxy"] = np.where(
        out["intervention_cost_proxy"] > 0,
        out["net_save_value_proxy"] / out["intervention_cost_proxy"],
        np.nan,
    )

    return out.sort_values("net_save_value_proxy", ascending=False)


def make_priority_tier_summary(scored: pd.DataFrame) -> pd.DataFrame:
    out = (
        scored.groupby("save_priority_tier", as_index=False)
        .agg(
            customers=("msno", "count"),
            avg_save_worthiness_score=("save_worthiness_score", "mean"),
            avg_predicted_churn_probability=("predicted_churn_probability", "mean"),
            actual_future_churn_rate=("actual_future_churn_label", "mean"),
            profit_adjusted_clv_proxy=("profit_adjusted_clv_proxy", "sum"),
            future_churned_clv_proxy=("future_churned_clv_proxy", "sum"),
            gross_value_at_risk_proxy=("gross_value_at_risk_proxy", "sum"),
            expected_saved_clv_proxy=("expected_saved_clv_proxy", "sum"),
            intervention_cost_proxy=("intervention_cost_proxy", "sum"),
            net_save_value_proxy=("net_save_value_proxy", "sum"),
            paid_offer_customers=("should_receive_paid_offer", "sum"),
        )
    )

    out["roi_proxy"] = np.where(
        out["intervention_cost_proxy"] > 0,
        out["net_save_value_proxy"] / out["intervention_cost_proxy"],
        np.nan,
    )

    return out.sort_values("net_save_value_proxy", ascending=False)


def make_budget_summary(scored: pd.DataFrame) -> pd.DataFrame:
    out = (
        scored.groupby("retention_budget_tier", as_index=False)
        .agg(
            customers=("msno", "count"),
            avg_predicted_churn_probability=("predicted_churn_probability", "mean"),
            actual_future_churn_rate=("actual_future_churn_label", "mean"),
            profit_adjusted_clv_proxy=("profit_adjusted_clv_proxy", "sum"),
            future_churned_clv_proxy=("future_churned_clv_proxy", "sum"),
            expected_saved_clv_proxy=("expected_saved_clv_proxy", "sum"),
            intervention_cost_proxy=("intervention_cost_proxy", "sum"),
            net_save_value_proxy=("net_save_value_proxy", "sum"),
            paid_offer_customers=("should_receive_paid_offer", "sum"),
        )
    )

    out["roi_proxy"] = np.where(
        out["intervention_cost_proxy"] > 0,
        out["net_save_value_proxy"] / out["intervention_cost_proxy"],
        np.nan,
    )

    return out.sort_values("net_save_value_proxy", ascending=False)


def make_segment_summary(scored: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "lifecycle_stage",
        "engagement_tier",
        "revenue_tier",
        "clv_value_tier",
        "churn_risk_tier",
        "retention_budget_tier",
        "recommended_retention_action",
        "save_priority_tier",
    ]

    out = (
        scored.groupby(group_cols, as_index=False)
        .agg(
            customers=("msno", "count"),
            avg_save_worthiness_score=("save_worthiness_score", "mean"),
            avg_predicted_churn_probability=("predicted_churn_probability", "mean"),
            actual_future_churn_rate=("actual_future_churn_label", "mean"),
            profit_adjusted_clv_proxy=("profit_adjusted_clv_proxy", "sum"),
            future_churned_clv_proxy=("future_churned_clv_proxy", "sum"),
            gross_value_at_risk_proxy=("gross_value_at_risk_proxy", "sum"),
            expected_saved_clv_proxy=("expected_saved_clv_proxy", "sum"),
            intervention_cost_proxy=("intervention_cost_proxy", "sum"),
            net_save_value_proxy=("net_save_value_proxy", "sum"),
            paid_offer_customers=("should_receive_paid_offer", "sum"),
        )
    )

    out = out[out["customers"] >= MIN_SEGMENT_SIZE].copy()

    out["roi_proxy"] = np.where(
        out["intervention_cost_proxy"] > 0,
        out["net_save_value_proxy"] / out["intervention_cost_proxy"],
        np.nan,
    )

    out["segment_strategy_readout"] = np.select(
        [
            out["save_priority_tier"].eq("Tier 1 - Executive save priority"),
            out["save_priority_tier"].eq("Tier 2 - Premium save priority"),
            out["recommended_retention_action"].eq("Suppress paid offer"),
            out["recommended_retention_action"].eq("Protect without discount"),
        ],
        [
            "Executive review segment",
            "Premium campaign candidate",
            "Suppression segment",
            "Protect relationship without paid offer",
        ],
        default="Monitor or test",
    )

    return out.sort_values("net_save_value_proxy", ascending=False)


def make_save_worthiness_deciles(scored: pd.DataFrame) -> pd.DataFrame:
    eligible = scored[scored["should_receive_paid_offer"]].copy()

    if eligible.empty:
        return pd.DataFrame()

    eligible["save_worthiness_decile"] = pd.qcut(
        eligible["save_worthiness_score"].rank(method="first", ascending=False),
        q=10,
        labels=list(range(1, 11)),
    ).astype(int)

    total_net = eligible["net_save_value_proxy"].sum()
    total_expected_saved = eligible["expected_saved_clv_proxy"].sum()

    out = (
        eligible.groupby("save_worthiness_decile", as_index=False)
        .agg(
            customers=("msno", "count"),
            avg_save_worthiness_score=("save_worthiness_score", "mean"),
            avg_predicted_churn_probability=("predicted_churn_probability", "mean"),
            actual_future_churn_rate=("actual_future_churn_label", "mean"),
            profit_adjusted_clv_proxy=("profit_adjusted_clv_proxy", "sum"),
            future_churned_clv_proxy=("future_churned_clv_proxy", "sum"),
            expected_saved_clv_proxy=("expected_saved_clv_proxy", "sum"),
            intervention_cost_proxy=("intervention_cost_proxy", "sum"),
            net_save_value_proxy=("net_save_value_proxy", "sum"),
        )
    ).sort_values("save_worthiness_decile")

    out["net_save_value_share"] = np.where(
        total_net > 0,
        out["net_save_value_proxy"] / total_net,
        0,
    )

    out["expected_saved_clv_share"] = np.where(
        total_expected_saved > 0,
        out["expected_saved_clv_proxy"] / total_expected_saved,
        0,
    )

    out["cumulative_net_save_value_share"] = out["net_save_value_share"].cumsum()
    out["cumulative_expected_saved_clv_share"] = out["expected_saved_clv_share"].cumsum()

    return out


def make_suppression_summary(scored: pd.DataFrame) -> pd.DataFrame:
    out = (
        scored.groupby("suppression_reason", as_index=False)
        .agg(
            customers=("msno", "count"),
            avg_predicted_churn_probability=("predicted_churn_probability", "mean"),
            actual_future_churn_rate=("actual_future_churn_label", "mean"),
            profit_adjusted_clv_proxy=("profit_adjusted_clv_proxy", "sum"),
            future_churned_clv_proxy=("future_churned_clv_proxy", "sum"),
            gross_value_at_risk_proxy=("gross_value_at_risk_proxy", "sum"),
            expected_saved_clv_proxy=("expected_saved_clv_proxy", "sum"),
            intervention_cost_proxy=("intervention_cost_proxy", "sum"),
            net_save_value_proxy=("net_save_value_proxy", "sum"),
        )
    )

    return out.sort_values("customers", ascending=False)


def validate_outputs(scored: pd.DataFrame, outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
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
        "scored_output_not_empty",
        len(scored),
        len(scored) > 0,
        "Save-worthiness scored customer output should not be empty.",
    )

    add_check(
        "one_row_per_customer",
        int(scored["msno"].duplicated().sum()),
        scored["msno"].duplicated().sum() == 0,
        "There should be one save-worthiness row per customer.",
    )

    add_check(
        "no_negative_scores",
        round(float(scored["save_worthiness_score"].min()), 6),
        scored["save_worthiness_score"].min() >= 0,
        "Save-worthiness score should be non-negative.",
    )

    add_check(
        "score_max_not_above_100",
        round(float(scored["save_worthiness_score"].max()), 6),
        scored["save_worthiness_score"].max() <= 100,
        "Save-worthiness score should be capped at 100.",
    )

    add_check(
        "positive_paid_offer_count",
        int(scored["should_receive_paid_offer"].sum()),
        int(scored["should_receive_paid_offer"].sum()) > 0,
        "There should be at least some economically positive paid-offer candidates.",
    )

    paid = scored[scored["should_receive_paid_offer"]]

    add_check(
        "paid_offer_net_value_positive",
        round(float(paid["net_save_value_proxy"].sum()), 2) if len(paid) > 0 else 0,
        len(paid) > 0 and paid["net_save_value_proxy"].sum() > 0,
        "Paid-offer candidate pool should have positive expected net value.",
    )

    add_check(
        "suppression_group_exists",
        int((scored["recommended_retention_action"] == "Suppress paid offer").sum()),
        int((scored["recommended_retention_action"] == "Suppress paid offer").sum()) > 0,
        "The engine should identify customers who should not receive paid offers.",
    )

    add_check(
        "decile_output_created",
        len(outputs["04_save_worthiness_deciles"]),
        len(outputs["04_save_worthiness_deciles"]) in [0, 10],
        "Save-worthiness decile output should have ten deciles if paid-offer candidates exist.",
    )

    report = pd.DataFrame(checks)
    report.to_csv(OUTPUT_DIR / "_save_worthiness_validation_report.csv", index=False)

    with open(OUTPUT_DIR / "_save_worthiness_validation_report.json", "w") as f:
        json.dump(clean_for_json(checks), f, indent=2)

    print("\nValidation report:")
    print(report.to_string(index=False))

    failed = report[report["status"] == "FAIL"]
    if len(failed) > 0:
        raise ValueError("Save-worthiness validation failed. Review outputs before moving forward.")

    return report


def write_sql_reference() -> None:
    sql_path = SQL_DIR / "08_save_worthiness_scoring.sql"

    lines = [
        "-- 08_save_worthiness_scoring.sql",
        "-- SQL reference for the save-worthiness decision engine.",
        "",
        "-- Core scoring idea:",
        "-- expected_saved_clv_proxy = predicted_churn_probability * profit_adjusted_clv_proxy * expected_response_rate",
        "-- net_save_value_proxy = expected_saved_clv_proxy - intervention_cost_proxy",
        "",
        "SELECT",
        "    msno,",
        "    predicted_churn_probability,",
        "    profit_adjusted_clv_proxy,",
        "    expected_intervention_response_rate,",
        "    intervention_cost_proxy,",
        "    expected_saved_clv_proxy,",
        "    net_save_value_proxy,",
        "    save_worthiness_score,",
        "    save_priority_tier,",
        "    recommended_retention_action",
        "FROM read_parquet('data/processed/save_worthiness_outputs/customer_save_worthiness_scores.parquet')",
        "ORDER BY net_save_value_proxy DESC;",
        "",
        "-- Recommended action summary",
        "SELECT",
        "    recommended_retention_action,",
        "    COUNT(*) AS customers,",
        "    SUM(expected_saved_clv_proxy) AS expected_saved_clv_proxy,",
        "    SUM(intervention_cost_proxy) AS intervention_cost_proxy,",
        "    SUM(net_save_value_proxy) AS net_save_value_proxy",
        "FROM read_parquet('data/processed/save_worthiness_outputs/customer_save_worthiness_scores.parquet')",
        "GROUP BY recommended_retention_action",
        "ORDER BY net_save_value_proxy DESC;",
    ]

    with open(sql_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved SQL reference file: {sql_path}")


def write_executive_summary(outputs: dict[str, pd.DataFrame], validation_report: pd.DataFrame) -> None:
    portfolio = outputs["00_save_worthiness_portfolio_summary"].iloc[0]
    action = outputs["01_retention_action_summary"].sort_values(
        "net_save_value_proxy",
        ascending=False,
    ).iloc[0]
    tier = outputs["02_save_priority_tier_summary"].sort_values(
        "net_save_value_proxy",
        ascending=False,
    ).iloc[0]
    segment = outputs["03_segment_save_worthiness_summary"].sort_values(
        "net_save_value_proxy",
        ascending=False,
    ).iloc[0]
    suppression = outputs["06_suppression_summary"].sort_values(
        "customers",
        ascending=False,
    ).iloc[0]

    summary = {
        "customers": int(portfolio["customers"]),
        "paid_offer_customers": int(portfolio["paid_offer_customers"]),
        "paid_offer_rate": float(portfolio["paid_offer_rate"]),
        "paid_offer_expected_saved_clv_proxy": float(portfolio["paid_offer_expected_saved_clv_proxy"]),
        "paid_offer_intervention_cost_proxy": float(portfolio["paid_offer_intervention_cost_proxy"]),
        "paid_offer_net_save_value_proxy": float(portfolio["paid_offer_net_save_value_proxy"]),
        "paid_offer_roi_proxy": (
            None
            if pd.isna(portfolio["paid_offer_roi_proxy"])
            else float(portfolio["paid_offer_roi_proxy"])
        ),
        "top_action": str(action["recommended_retention_action"]),
        "top_action_customers": int(action["customers"]),
        "top_action_net_save_value_proxy": float(action["net_save_value_proxy"]),
        "top_priority_tier": str(tier["save_priority_tier"]),
        "top_priority_tier_customers": int(tier["customers"]),
        "top_priority_tier_net_save_value_proxy": float(tier["net_save_value_proxy"]),
        "top_segment": {
            "lifecycle_stage": str(segment["lifecycle_stage"]),
            "engagement_tier": str(segment["engagement_tier"]),
            "revenue_tier": str(segment["revenue_tier"]),
            "clv_value_tier": str(segment["clv_value_tier"]),
            "churn_risk_tier": str(segment["churn_risk_tier"]),
            "retention_budget_tier": str(segment["retention_budget_tier"]),
            "recommended_retention_action": str(segment["recommended_retention_action"]),
            "customers": int(segment["customers"]),
            "net_save_value_proxy": float(segment["net_save_value_proxy"]),
        },
        "largest_suppression_reason": str(suppression["suppression_reason"]),
        "largest_suppression_customers": int(suppression["customers"]),
        "validation_status": "PASS" if (validation_report["status"] == "PASS").all() else "FAIL",
    }

    with open(OUTPUT_DIR / "_save_worthiness_summary.json", "w") as f:
        json.dump(clean_for_json(summary), f, indent=2)

    roi_line = (
        f"{summary['paid_offer_roi_proxy']:.2f}x"
        if summary["paid_offer_roi_proxy"] is not None
        else "N/A"
    )

    lines = [
        "# Save-Worthiness Scoring Summary",
        "",
        f"Customers scored: {summary['customers']:,}",
        f"Economically positive paid-offer candidates: {summary['paid_offer_customers']:,}",
        f"Paid-offer candidate rate: {summary['paid_offer_rate']:.2%}",
        f"Expected saved CLV proxy from paid-offer pool: {summary['paid_offer_expected_saved_clv_proxy']:,.0f}",
        f"Estimated intervention cost proxy for paid-offer pool: {summary['paid_offer_intervention_cost_proxy']:,.0f}",
        f"Expected net save value proxy: {summary['paid_offer_net_save_value_proxy']:,.0f}",
        f"Paid-offer ROI proxy: {roi_line}",
        "",
        "Key readout:",
        (
            f"- The highest expected-value action is **{summary['top_action']}**, with "
            f"{summary['top_action_customers']:,} customers and "
            f"{summary['top_action_net_save_value_proxy']:,.0f} expected net save value proxy."
        ),
        (
            f"- The strongest priority tier is **{summary['top_priority_tier']}**, with "
            f"{summary['top_priority_tier_customers']:,} customers and "
            f"{summary['top_priority_tier_net_save_value_proxy']:,.0f} expected net save value proxy."
        ),
        (
            f"- The top save-worthiness segment is **{summary['top_segment']['lifecycle_stage']} / "
            f"{summary['top_segment']['engagement_tier']} / "
            f"{summary['top_segment']['revenue_tier']} / "
            f"{summary['top_segment']['clv_value_tier']} / "
            f"{summary['top_segment']['churn_risk_tier']} / "
            f"{summary['top_segment']['retention_budget_tier']}**, with "
            f"{summary['top_segment']['customers']:,} customers."
        ),
        (
            f"- The largest suppression reason is **{summary['largest_suppression_reason']}**, covering "
            f"{summary['largest_suppression_customers']:,} customers."
        ),
        "",
        "Business interpretation:",
        (
            "This layer turns churn prediction into a retention decision engine. It avoids the common mistake "
            "of targeting customers only because they have high churn probability. The score prioritizes customers "
            "where churn risk, customer value, expected response, and intervention cost create positive expected economics."
        ),
        "",
        "Next step:",
        (
            "The next script should run retention ROI simulation. It should stress-test response rates, discount costs, "
            "campaign budgets, and targeting thresholds before recommending a campaign strategy."
        ),
        "",
        "Assumption note:",
        (
            "Response rates and intervention costs are planning assumptions. They are intentionally separated into this "
            "layer so the ROI simulator can test conservative, base, and aggressive scenarios later."
        ),
    ]

    summary_path = OUTPUT_DIR / "_executive_save_worthiness_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved {summary_path}")
    print(f"Saved {OUTPUT_DIR / '_save_worthiness_summary.json'}")


def main() -> None:
    print("\nRunning save-worthiness scoring engine...")

    with open(OUTPUT_DIR / "_save_worthiness_assumptions.json", "w") as f:
        json.dump(ASSUMPTIONS, f, indent=2)

    base = load_inputs()
    scored = recommend_intervention_action(base)
    scored = attach_assumptions_and_scores(scored)

    output_columns = [
        "msno",
        "snapshot_month",
        "actual_future_churn_label",
        "predicted_churn_probability",
        "churn_risk_decile",
        "churn_risk_tier",
        "lifecycle_stage",
        "engagement_tier",
        "revenue_tier",
        "clv_value_tier",
        "value_risk_quadrant",
        "lifecycle_strategy_readout",
        "value_based_action_group",
        "retention_budget_tier",
        "monthly_value_baseline",
        "monthly_margin_baseline",
        "annual_revenue_run_rate_proxy",
        "annual_margin_run_rate_proxy",
        "profit_adjusted_clv_proxy",
        "future_churned_clv_proxy",
        "gross_value_at_risk_proxy",
        "expected_intervention_response_rate",
        "intervention_cost_proxy",
        "expected_saved_clv_proxy",
        "net_save_value_proxy",
        "break_even_response_rate",
        "expected_roi_proxy",
        "recommended_retention_action",
        "should_receive_paid_offer",
        "suppression_reason",
        "save_worthiness_rank",
        "save_worthiness_score",
        "save_priority_tier",
        "final_targeting_readout",
    ]

    scored = scored[[col for col in output_columns if col in scored.columns]].copy()

    outputs = {
        "00_save_worthiness_portfolio_summary": make_portfolio_summary(scored),
        "01_retention_action_summary": make_action_summary(scored),
        "02_save_priority_tier_summary": make_priority_tier_summary(scored),
        "03_segment_save_worthiness_summary": make_segment_summary(scored),
        "04_save_worthiness_deciles": make_save_worthiness_deciles(scored),
        "05_retention_budget_summary": make_budget_summary(scored),
        "06_suppression_summary": make_suppression_summary(scored),
    }

    write_sql_reference()

    scored.to_parquet(OUTPUT_DIR / "customer_save_worthiness_scores.parquet", index=False)
    # The parquet file is the source of truth. CSV samples are exported for easier
    # inspection and Tableau development without loading the full customer table.
    scored.head(100000).to_csv(
        OUTPUT_DIR / "tableau_save_worthiness_customer_sample.csv",
        index=False,
    )

    paid_targets = scored[scored["should_receive_paid_offer"]].copy()
    paid_targets.head(100000).to_csv(
        OUTPUT_DIR / "high_priority_save_targets.csv",
        index=False,
    )

    tableau_base = outputs["03_segment_save_worthiness_summary"].copy()
    tableau_base.to_csv(
        OUTPUT_DIR / "tableau_save_worthiness_segment_base.csv",
        index=False,
    )

    print(f"Saved {OUTPUT_DIR / 'customer_save_worthiness_scores.parquet'}")
    print(f"Saved {OUTPUT_DIR / 'tableau_save_worthiness_customer_sample.csv'}")
    print(f"Saved {OUTPUT_DIR / 'high_priority_save_targets.csv'}")
    print(f"Saved {OUTPUT_DIR / 'tableau_save_worthiness_segment_base.csv'}")

    for name, df in outputs.items():
        output_path = OUTPUT_DIR / f"{name}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path} | rows={len(df):,} cols={df.shape[1]:,}")

    validation_report = validate_outputs(scored, outputs)
    write_executive_summary(outputs, validation_report)

    print("\n08_save_worthiness_scoring.py complete.")


if __name__ == "__main__":
    main()
