"""
10_ab_test_planner.py

A/B test planning layer for the Customer Revenue Recovery & Retention ROI Engine.

The previous scripts built the decision engine:

05_churn_model.py
    Who is likely to leave?

08_save_worthiness_scoring.py
    Who is economically worth saving?

09_retention_roi_simulation.py
    Which campaign strategy and budget frontier look attractive?

This script answers the last question before launch:

How do we prove the retention strategy actually works?

This is not a campaign result script.
It is an experiment design script.

It builds:
1. A recommended A/B test population
2. A treatment/control assignment plan
3. Primary and secondary success metrics
4. Sample-size and power scenarios
5. Break-even and rollout decision rules
6. Tableau-ready experiment planning outputs

Launch-readiness note:
The goal is not to make a retention campaign look good on paper.
The goal is to give leadership a test that can kill the idea if the economics do not hold.
"""

from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed")
SAVE_DIR = PROCESSED_DIR / "save_worthiness_outputs"
ROI_DIR = PROCESSED_DIR / "retention_roi_outputs"
OUTPUT_DIR = PROCESSED_DIR / "ab_test_outputs"
SQL_DIR = Path("sql")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SQL_DIR.mkdir(parents=True, exist_ok=True)

SAVE_SCORES_PATH = SAVE_DIR / "customer_save_worthiness_scores.parquet"
ROI_MATRIX_PATH = ROI_DIR / "01_strategy_scenario_matrix.csv"
BUDGET_FRONTIER_PATH = ROI_DIR / "03_budget_cap_frontier.csv"
BEST_STRATEGY_PATH = ROI_DIR / "05_best_strategy_by_scenario.csv"

# The assignment needs to be reproducible. A fixed seed lets the same customer
# receive the same treatment/control assignment every time the script runs.
RANDOMIZATION_SEED = "customer_revenue_recovery_ab_test_v1"
PRIMARY_ALPHA = 0.05
DEFAULT_TREATMENT_SHARE = 0.50

# Z constants avoid adding a scipy dependency.
# These power calculations are planning approximations, not a replacement for
# a full statistical testing package in production.
Z_ALPHA_05_TWO_SIDED = 1.96
Z_POWER = {
    0.80: 0.84,
    0.90: 1.28,
    0.95: 1.645,
}

RESPONSE_MULTIPLIER_SCENARIOS = [
    ("Half expected response", 0.50),
    ("Conservative expected response", 0.70),
    ("Base expected response", 1.00),
    ("Upside expected response", 1.25),
]

REQUIRED_SAVE_COLUMNS = [
    "msno",
    "actual_future_churn_label",
    "predicted_churn_probability",
    "profit_adjusted_clv_proxy",
    "gross_value_at_risk_proxy",
    "expected_intervention_response_rate",
    "intervention_cost_proxy",
    "expected_saved_clv_proxy",
    "net_save_value_proxy",
    "recommended_retention_action",
    "should_receive_paid_offer",
    "save_worthiness_score",
    "save_priority_tier",
    "final_targeting_readout",
    "churn_risk_tier",
    "clv_value_tier",
    "lifecycle_stage",
    "engagement_tier",
    "revenue_tier",
    "retention_budget_tier",
    "future_churned_clv_proxy",
]


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


def require_file(path, upstream_script):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run {upstream_script} first.")


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def two_proportion_sample_size(p_control, p_treatment, alpha=0.05, power=0.80):
    p_control = float(np.clip(p_control, 1e-6, 1 - 1e-6))
    p_treatment = float(np.clip(p_treatment, 1e-6, 1 - 1e-6))

    effect = abs(p_control - p_treatment)
    if effect <= 0:
        return np.inf

    z_alpha = Z_ALPHA_05_TWO_SIDED if alpha == 0.05 else Z_ALPHA_05_TWO_SIDED
    z_beta = Z_POWER.get(round(power, 2), 0.84)

    pooled = (p_control + p_treatment) / 2

    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_beta * math.sqrt(
            p_control * (1 - p_control)
            + p_treatment * (1 - p_treatment)
        )
    ) ** 2

    return math.ceil(numerator / (effect ** 2))


def approximate_power_two_proportion(p_control, p_treatment, n_per_group, alpha=0.05):
    p_control = float(np.clip(p_control, 1e-6, 1 - 1e-6))
    p_treatment = float(np.clip(p_treatment, 1e-6, 1 - 1e-6))
    n_per_group = max(int(n_per_group), 1)

    effect = abs(p_control - p_treatment)
    pooled = (p_control + p_treatment) / 2

    null_se = math.sqrt(2 * pooled * (1 - pooled) / n_per_group)
    alt_se = math.sqrt(
        (
            p_control * (1 - p_control)
            + p_treatment * (1 - p_treatment)
        )
        / n_per_group
    )

    if null_se <= 0 or alt_se <= 0:
        return np.nan

    z_alpha = Z_ALPHA_05_TWO_SIDED if alpha == 0.05 else Z_ALPHA_05_TWO_SIDED
    z_effect = (effect - z_alpha * null_se) / alt_se

    return float(np.clip(normal_cdf(z_effect), 0, 1))

# Hash-based randomization makes assignment stable without storing a separate
# random number file. This is useful for a portfolio pipeline because reruns
# should not reshuffle customers.
def stable_random_number(value, seed=RANDOMIZATION_SEED):
    raw = f"{seed}|{value}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return int(digest, 16) / float(16 ** 16 - 1)


def load_inputs():
    require_file(SAVE_SCORES_PATH, "src/08_save_worthiness_scoring.py")
    require_file(ROI_MATRIX_PATH, "src/09_retention_roi_simulation.py")
    require_file(BUDGET_FRONTIER_PATH, "src/09_retention_roi_simulation.py")
    require_file(BEST_STRATEGY_PATH, "src/09_retention_roi_simulation.py")

    print("\nLoading save-worthiness and ROI simulation outputs...")

    save_scores = pd.read_parquet(SAVE_SCORES_PATH)
    roi_matrix = pd.read_csv(ROI_MATRIX_PATH)
    budget_frontier = pd.read_csv(BUDGET_FRONTIER_PATH)
    best_strategy = pd.read_csv(BEST_STRATEGY_PATH)

    missing = sorted(set(REQUIRED_SAVE_COLUMNS) - set(save_scores.columns))
    if missing:
        raise ValueError(f"Missing required columns from save-worthiness scores: {missing}")

    numeric_cols = [
        "actual_future_churn_label",
        "predicted_churn_probability",
        "profit_adjusted_clv_proxy",
        "gross_value_at_risk_proxy",
        "expected_intervention_response_rate",
        "intervention_cost_proxy",
        "expected_saved_clv_proxy",
        "net_save_value_proxy",
        "save_worthiness_score",
        "future_churned_clv_proxy",
    ]

    for col in numeric_cols:
        save_scores[col] = pd.to_numeric(save_scores[col], errors="coerce").fillna(0)

    # Actual future churn is retrospective only. It supports validation and
    # balance diagnostics, but the planned targeting decision is based on
    # predicted churn risk, CLV, and expected campaign economics.
    save_scores["actual_future_churn_label"] = save_scores["actual_future_churn_label"].astype(int)
    save_scores["should_receive_paid_offer"] = save_scores["should_receive_paid_offer"].astype(bool)

    print(f"Loaded customers: {len(save_scores):,}")
    print(f"Paid-offer candidates: {save_scores['should_receive_paid_offer'].sum():,}")
    print(f"ROI strategy/scenario rows: {len(roi_matrix):,}")

    return save_scores, roi_matrix, budget_frontier, best_strategy


def choose_efficient_budget_frontier(budget_frontier):
    # Use the same efficient-frontier idea from the ROI simulator:
    # choose the smallest tested budget that captures at least 99% of the
    # maximum tested net value.
    frontier = budget_frontier.copy()
    frontier["net_save_value_proxy"] = pd.to_numeric(
        frontier["net_save_value_proxy"],
        errors="coerce",
    ).fillna(0)

    max_net_value = frontier["net_save_value_proxy"].max()

    if max_net_value <= 0:
        selected = frontier.sort_values("budget_cap").iloc[0].copy()
        selected["net_value_capture_vs_max"] = 0
        return selected

    frontier["net_value_capture_vs_max"] = frontier["net_save_value_proxy"] / max_net_value

    selected = (
        frontier[frontier["net_value_capture_vs_max"] >= 0.99]
        .sort_values("budget_cap")
        .iloc[0]
    )

    return selected


def build_experiment_population(save_scores, budget_frontier):
    efficient_budget = choose_efficient_budget_frontier(budget_frontier)

    # Start with customers who are economically positive paid-offer candidates.
    # The experiment should test the recommended strategy, not a broad churn-risk list.
    eligible = save_scores[
        save_scores["should_receive_paid_offer"]
        & save_scores["net_save_value_proxy"].gt(0)
        & save_scores["intervention_cost_proxy"].gt(0)
    ].copy()

    eligible = eligible.sort_values(
        ["net_save_value_proxy", "save_worthiness_score", "predicted_churn_probability"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # Rank by expected net value, then add customers until the efficient budget
    # cap is reached. This keeps the test aligned with the ROI frontier.
    eligible["rollout_cumulative_cost_proxy"] = eligible["intervention_cost_proxy"].cumsum()

    budget_cap = float(efficient_budget["budget_cap"])

    experiment_population = eligible[
        eligible["rollout_cumulative_cost_proxy"] <= budget_cap
    ].copy()

    if experiment_population.empty:
        experiment_population = eligible.head(5000).copy()

    # Stratification keeps treatment and control comparable across the segments
    # that matter most for interpretation: priority, risk, value, and lifecycle.
    stratification_cols = [
        "save_priority_tier",
        "churn_risk_tier",
        "clv_value_tier",
        "lifecycle_stage",
    ]

    experiment_population["randomization_stratum"] = (
        experiment_population[stratification_cols]
        .fillna("Unknown")
        .astype(str)
        .agg(" | ".join, axis=1)
    )

    # The randomization score is deterministic, so the assignment can be audited
    # and recreated without relying on a one-time random draw.
    experiment_population["ab_test_randomization_score"] = (
        experiment_population["msno"]
        .astype(str)
        .map(lambda value: stable_random_number(value))
    )

    experiment_population = experiment_population.sort_values(
        ["randomization_stratum", "ab_test_randomization_score"]
    ).reset_index(drop=True)

    experiment_population["stratum_row_number"] = (
        experiment_population.groupby("randomization_stratum").cumcount() + 1
    )

    experiment_population["stratum_size"] = (
        experiment_population.groupby("randomization_stratum")["msno"].transform("count")
    )

    experiment_population["stratum_treatment_cutoff"] = np.ceil(
        experiment_population["stratum_size"] * DEFAULT_TREATMENT_SHARE
    ).astype(int)

    experiment_population["experiment_group"] = np.where(
        experiment_population["stratum_row_number"] <= experiment_population["stratum_treatment_cutoff"],
        "Treatment - retention offer",
        "Control - holdout",
    )

    experiment_population["experiment_name"] = "Save-worthy retention offer A/B test"
    experiment_population["randomization_unit"] = "Customer account / msno"
    experiment_population["planned_rollout_budget_cap"] = budget_cap
    experiment_population["efficient_budget_value_capture_vs_max"] = float(
        efficient_budget.get("net_value_capture_vs_max", np.nan)
    )
    experiment_population["randomization_method"] = (
        "Deterministic stratified randomization using save priority, churn risk, CLV tier, and lifecycle stage."
    )

    return experiment_population, efficient_budget


def make_ab_test_portfolio_summary(save_scores, experiment_population, efficient_budget):
    paid_offer = save_scores["should_receive_paid_offer"]
    treatment = experiment_population["experiment_group"].eq("Treatment - retention offer")

    row = {
        "total_customers_scored": int(len(save_scores)),
        "paid_offer_candidates_from_save_worthiness": int(paid_offer.sum()),
        "ab_test_candidate_customers": int(len(experiment_population)),
        "treatment_customers": int(treatment.sum()),
        "control_customers": int((~treatment).sum()),
        "recommended_rollout_budget_cap_proxy": float(efficient_budget["budget_cap"]),
        "rollout_population_cost_proxy": float(experiment_population["intervention_cost_proxy"].sum()),
        "efficient_budget_value_capture_vs_max": float(
            experiment_population["efficient_budget_value_capture_vs_max"].iloc[0]
        ),
        "avg_predicted_churn_probability": float(
            experiment_population["predicted_churn_probability"].mean()
        ),
        "avg_expected_intervention_response_rate": float(
            experiment_population["expected_intervention_response_rate"].mean()
        ),
        "actual_future_churn_rate_retrospective": float(
            experiment_population["actual_future_churn_label"].mean()
        ),
    }

    return pd.DataFrame([row])


def make_experiment_design_summary(experiment_population):
    treatment = experiment_population[
        experiment_population["experiment_group"].eq("Treatment - retention offer")
    ].copy()

    control = experiment_population[
        experiment_population["experiment_group"].eq("Control - holdout")
    ].copy()

    p_control = float(control["predicted_churn_probability"].mean())

    # This is the planning treatment churn rate, not an observed result.
    # The actual treatment effect would come from the live experiment readout.
    expected_treatment_churn_rate = float(
        (
            treatment["predicted_churn_probability"]
            * (1 - treatment["expected_intervention_response_rate"])
        ).mean()
    )

    expected_absolute_churn_reduction = p_control - expected_treatment_churn_rate

    treatment_cost = float(treatment["intervention_cost_proxy"].sum())
    treatment_gross_value_at_risk = float(treatment["gross_value_at_risk_proxy"].sum())
    treatment_expected_saved_clv = float(treatment["expected_saved_clv_proxy"].sum())
    treatment_net_value = treatment_expected_saved_clv - treatment_cost
    # Business success is measured after campaign cost, so a churn reduction
    # alone is not enough to justify rollout.

    expected_customers_saved = float(
        (
            treatment["predicted_churn_probability"]
            * treatment["expected_intervention_response_rate"]
        ).sum()
    )

    break_even_response_rate = (
        treatment_cost / treatment_gross_value_at_risk
        if treatment_gross_value_at_risk > 0
        else np.nan
    )

    roi_proxy = (
        treatment_net_value / treatment_cost
        if treatment_cost > 0
        else np.nan
    )

    n_per_group = min(len(treatment), len(control))
    approximate_power = approximate_power_two_proportion(
        p_control=p_control,
        p_treatment=max(expected_treatment_churn_rate, 1e-6),
        n_per_group=n_per_group,
        alpha=PRIMARY_ALPHA,
    )

    row = {
        "experiment_name": "Save-worthy retention offer A/B test",
        "experiment_goal": "Measure whether save-worthy retention targeting reduces churn and creates positive incremental net value.",
        "randomization_unit": "Customer account / msno",
        "primary_metric": "Incremental churn reduction",
        "primary_business_metric": "Incremental net save value proxy",
        "candidate_customers": int(len(experiment_population)),
        "treatment_customers": int(len(treatment)),
        "control_customers": int(len(control)),
        "treatment_share": float(len(treatment) / len(experiment_population)) if len(experiment_population) > 0 else np.nan,
        "control_churn_rate_planning": p_control,
        "treatment_churn_rate_planning": expected_treatment_churn_rate,
        "expected_absolute_churn_reduction": expected_absolute_churn_reduction,
        "expected_relative_churn_reduction": (
            expected_absolute_churn_reduction / p_control
            if p_control > 0
            else np.nan
        ),
        "expected_customers_saved_in_treatment": expected_customers_saved,
        "treatment_expected_saved_clv_proxy": treatment_expected_saved_clv,
        "treatment_campaign_cost_proxy": treatment_cost,
        "treatment_expected_net_save_value_proxy": treatment_net_value,
        "treatment_roi_proxy": roi_proxy,
        "break_even_response_rate": break_even_response_rate,
        "approx_power_for_base_effect": approximate_power,
        "planning_alpha": PRIMARY_ALPHA,
        "test_recommendation": (
            "Proceed to controlled A/B test"
            if treatment_net_value > 0 and approximate_power >= 0.80
            else "Proceed only after reviewing effect size, power, or economics"
        ),
    }

    return pd.DataFrame([row])


def make_power_sample_size_scenarios(experiment_population):
    rows = []

    p_control = float(experiment_population["predicted_churn_probability"].mean())
    planned_per_group = int(len(experiment_population) // 2)

    for scenario_label, response_multiplier in RESPONSE_MULTIPLIER_SCENARIOS:
        adjusted_response = (
            experiment_population["expected_intervention_response_rate"]
            * response_multiplier
        ).clip(lower=0, upper=0.60)

        p_treatment = float(
            (
                experiment_population["predicted_churn_probability"]
                * (1 - adjusted_response)
            ).mean()
        )

        absolute_effect = p_control - p_treatment
        relative_effect = absolute_effect / p_control if p_control > 0 else np.nan

        for target_power in [0.80, 0.90, 0.95]:
            required_per_group = two_proportion_sample_size(
                p_control=p_control,
                p_treatment=max(p_treatment, 1e-6),
                alpha=PRIMARY_ALPHA,
                power=target_power,
            )

            actual_power = approximate_power_two_proportion(
                p_control=p_control,
                p_treatment=max(p_treatment, 1e-6),
                n_per_group=planned_per_group,
                alpha=PRIMARY_ALPHA,
            )

            rows.append(
                {
                    "scenario_label": scenario_label,
                    "response_multiplier": response_multiplier,
                    "control_churn_rate_planning": p_control,
                    "treatment_churn_rate_planning": p_treatment,
                    "absolute_churn_reduction": absolute_effect,
                    "relative_churn_reduction": relative_effect,
                    "target_power": target_power,
                    "alpha": PRIMARY_ALPHA,
                    "required_sample_size_per_group": (
                        None if math.isinf(required_per_group) else int(required_per_group)
                    ),
                    "planned_sample_size_per_group": planned_per_group,
                    "planned_sample_size_total": int(len(experiment_population)),
                    "approx_power_with_planned_sample": actual_power,
                    "power_readout": (
                        "Planned sample should be sufficient"
                        if actual_power >= target_power
                        else "Planned sample may be underpowered for this scenario"
                    ),
                }
            )

    return pd.DataFrame(rows)


def make_treatment_control_plan(experiment_population):
    out = (
        experiment_population.groupby("experiment_group", as_index=False)
        .agg(
            customers=("msno", "count"),
            avg_randomization_score=("ab_test_randomization_score", "mean"),
            avg_predicted_churn_probability=("predicted_churn_probability", "mean"),
            actual_future_churn_rate_retrospective=("actual_future_churn_label", "mean"),
            profit_adjusted_clv_proxy=("profit_adjusted_clv_proxy", "sum"),
            future_churned_clv_proxy=("future_churned_clv_proxy", "sum"),
            gross_value_at_risk_proxy=("gross_value_at_risk_proxy", "sum"),
            expected_saved_clv_proxy=("expected_saved_clv_proxy", "sum"),
            intervention_cost_proxy=("intervention_cost_proxy", "sum"),
            net_save_value_proxy=("net_save_value_proxy", "sum"),
        )
    )

    out["planned_campaign_cost_proxy"] = np.where(
        out["experiment_group"].eq("Treatment - retention offer"),
        out["intervention_cost_proxy"],
        0,
    )

    out["group_role"] = np.where(
        out["experiment_group"].eq("Treatment - retention offer"),
        "Receives assigned retention intervention",
        "Holdout group for causal measurement",
    )

    return out.sort_values("experiment_group", ascending=False)


def make_metric_definitions():
    rows = [
        {
            "metric_name": "Incremental churn reduction",
            "metric_type": "Primary statistical metric",
            "definition": "Control churn rate minus treatment churn rate over the measurement window.",
            "direction": "Higher is better",
            "why_it_matters": "Shows whether the offer actually reduces churn versus a holdout.",
        },
        {
            "metric_name": "Incremental net save value proxy",
            "metric_type": "Primary business metric",
            "definition": "Incremental saved CLV proxy minus treatment campaign cost.",
            "direction": "Higher is better",
            "why_it_matters": "Prevents a campaign from being judged successful only because churn fell.",
        },
        {
            "metric_name": "Break-even response rate",
            "metric_type": "Economic guardrail",
            "definition": "Treatment campaign cost divided by gross value at risk proxy.",
            "direction": "Lower is better",
            "why_it_matters": "Shows the minimum response needed for the campaign to pay for itself.",
        },
        {
            "metric_name": "Treatment ROI proxy",
            "metric_type": "Business guardrail",
            "definition": "Treatment net save value proxy divided by treatment campaign cost proxy.",
            "direction": "Higher is better",
            "why_it_matters": "Summarizes whether the campaign is worth scaling.",
        },
        {
            "metric_name": "Customer experience guardrail",
            "metric_type": "Operational guardrail",
            "definition": "Monitor complaint rate, support contact rate, refund pressure, and offer abuse.",
            "direction": "Lower is better",
            "why_it_matters": "A profitable-looking campaign can still damage customer experience.",
        },
        {
            "metric_name": "Offer cannibalization check",
            "metric_type": "Commercial guardrail",
            "definition": "Track whether customers who would have stayed anyway are taking unnecessary discounts.",
            "direction": "Lower is better",
            "why_it_matters": "Protects margin from over-discounting.",
        },
    ]

    return pd.DataFrame(rows)


def make_randomization_unit_plan(experiment_population):
    treatment_count = int(
        experiment_population["experiment_group"].eq("Treatment - retention offer").sum()
    )
    control_count = int(
        experiment_population["experiment_group"].eq("Control - holdout").sum()
    )

    rows = [
        {
            "plan_component": "Randomization unit",
            "value": "Customer account / msno",
            "readout": "Randomize at the customer level so one customer cannot appear in both groups.",
        },
        {
            "plan_component": "Randomization seed",
            "value": RANDOMIZATION_SEED,
            "readout": "Stable seed makes the assignment reproducible.",
        },
        {
            "plan_component": "Treatment group",
            "value": treatment_count,
            "readout": "Customers assigned to receive the recommended retention intervention.",
        },
        {
            "plan_component": "Control group",
            "value": control_count,
            "readout": "Holdout customers used to measure incremental lift.",
        },
        {
            "plan_component": "Balance check",
            "value": f"{treatment_count / max(len(experiment_population), 1):.2%} treatment share",
            "readout": "Near 50/50 allocation keeps the first test simple and defensible.",
        },
        {
            "plan_component": "Stratification note",
            "value": "Monitor balance by save priority tier, churn risk tier, CLV tier, and lifecycle stage.",
            "readout": "The output includes segment readiness checks to verify group balance.",
        },
    ]

    return pd.DataFrame(rows)


def make_rollout_decision_rules(design_summary):
    design = design_summary.iloc[0]

    break_even = float(design["break_even_response_rate"])
    expected_abs_lift = float(design["expected_absolute_churn_reduction"])

    rows = [
        {
            "decision_stage": "Launch test",
            "rule_name": "Population quality",
            "decision_rule": "Use the efficient-frontier save-worthy population rather than a blanket churn-risk list.",
            "pass_condition": "Candidate pool has positive expected net value and valid treatment/control split.",
        },
        {
            "decision_stage": "Readout",
            "rule_name": "Primary statistical success",
            "decision_rule": "Treatment churn rate should be lower than control churn rate.",
            "pass_condition": f"Target planning lift is about {expected_abs_lift:.2%} absolute churn reduction.",
        },
        {
            "decision_stage": "Readout",
            "rule_name": "Business success",
            "decision_rule": "Incremental saved CLV proxy should exceed campaign cost.",
            "pass_condition": "Observed net save value proxy is positive.",
        },
        {
            "decision_stage": "Readout",
            "rule_name": "Break-even response",
            "decision_rule": "Observed response should clear break-even economics.",
            "pass_condition": f"Observed response rate exceeds approximately {break_even:.2%}.",
        },
        {
            "decision_stage": "Scale",
            "rule_name": "Scale decision",
            "decision_rule": "Scale only if statistical lift, positive net value, and customer-experience guardrails all pass.",
            "pass_condition": "Do not scale from churn reduction alone.",
        },
        {
            "decision_stage": "Stop",
            "rule_name": "Kill condition",
            "decision_rule": "Stop or redesign if lift is weak, economics are negative, or complaints/refunds rise.",
            "pass_condition": "A failed test is a valid result if it prevents bad retention spend.",
        },
    ]

    return pd.DataFrame(rows)


def make_segment_test_readiness(experiment_population):
    group_cols = [
        "lifecycle_stage",
        "engagement_tier",
        "revenue_tier",
        "clv_value_tier",
        "churn_risk_tier",
        "save_priority_tier",
    ]

    out = (
        experiment_population.groupby(group_cols, as_index=False)
        .agg(
            customers=("msno", "count"),
            treatment_customers=("experiment_group", lambda s: int((s == "Treatment - retention offer").sum())),
            control_customers=("experiment_group", lambda s: int((s == "Control - holdout").sum())),
            avg_predicted_churn_probability=("predicted_churn_probability", "mean"),
            actual_future_churn_rate_retrospective=("actual_future_churn_label", "mean"),
            gross_value_at_risk_proxy=("gross_value_at_risk_proxy", "sum"),
            expected_saved_clv_proxy=("expected_saved_clv_proxy", "sum"),
            intervention_cost_proxy=("intervention_cost_proxy", "sum"),
            net_save_value_proxy=("net_save_value_proxy", "sum"),
        )
    )

    out["treatment_share"] = out["treatment_customers"] / out["customers"]
    out["balance_status"] = np.select(
        [
            out["customers"] < 100,
            out["treatment_share"].between(0.45, 0.55),
            ~out["treatment_share"].between(0.45, 0.55),
        ],
        [
            "Small segment; directional only",
            "Balanced",
            "Review balance",
        ],
        default="Review",
    )

    out["test_readiness_readout"] = np.select(
        [
            (out["customers"] >= 500) & (out["balance_status"].eq("Balanced")),
            out["customers"] >= 100,
        ],
        [
            "Strong segment readout candidate",
            "Usable directional readout",
        ],
        default="Too small for standalone readout",
    )

    return out.sort_values("net_save_value_proxy", ascending=False)


def make_tableau_ab_test_plan(portfolio_summary, design_summary, treatment_plan, power_scenarios):
    design = design_summary.iloc[0]
    portfolio = portfolio_summary.iloc[0]

    base_power = power_scenarios[
        (power_scenarios["scenario_label"] == "Base expected response")
        & (power_scenarios["target_power"] == 0.80)
    ]

    if base_power.empty:
        base_power_row = power_scenarios.iloc[0]
    else:
        base_power_row = base_power.iloc[0]

    rows = [
        {
            "section": "Experiment overview",
            "metric": "Candidate customers",
            "value": float(design["candidate_customers"]),
            "display_value": f"{int(design['candidate_customers']):,}",
            "readout": "Efficient-frontier save-worthy population selected for testing.",
        },
        {
            "section": "Experiment overview",
            "metric": "Treatment customers",
            "value": float(design["treatment_customers"]),
            "display_value": f"{int(design['treatment_customers']):,}",
            "readout": "Customers assigned to receive retention intervention.",
        },
        {
            "section": "Experiment overview",
            "metric": "Control customers",
            "value": float(design["control_customers"]),
            "display_value": f"{int(design['control_customers']):,}",
            "readout": "Holdout customers for causal measurement.",
        },
        {
            "section": "Expected impact",
            "metric": "Expected customers saved in treatment",
            "value": float(design["expected_customers_saved_in_treatment"]),
            "display_value": f"{design['expected_customers_saved_in_treatment']:,.0f}",
            "readout": "Planning estimate before A/B validation.",
        },
        {
            "section": "Expected impact",
            "metric": "Treatment expected net save value proxy",
            "value": float(design["treatment_expected_net_save_value_proxy"]),
            "display_value": f"{design['treatment_expected_net_save_value_proxy']:,.0f}",
            "readout": "Expected net value for the treatment arm only.",
        },
        {
            "section": "Expected impact",
            "metric": "Treatment ROI proxy",
            "value": float(design["treatment_roi_proxy"]),
            "display_value": f"{design['treatment_roi_proxy']:.2f}x",
            "readout": "Planning ROI for test treatment arm.",
        },
        {
            "section": "Power",
            "metric": "Approximate power for base effect",
            "value": float(design["approx_power_for_base_effect"]),
            "display_value": f"{design['approx_power_for_base_effect']:.2%}",
            "readout": "Approximate two-proportion power for planned churn reduction.",
        },
        {
            "section": "Power",
            "metric": "Required sample per group at 80% power",
            "value": float(base_power_row["required_sample_size_per_group"]),
            "display_value": f"{int(base_power_row['required_sample_size_per_group']):,}",
            "readout": "Planning requirement for the base expected response scenario.",
        },
        {
            "section": "Budget",
            "metric": "Recommended rollout budget cap proxy",
            "value": float(portfolio["recommended_rollout_budget_cap_proxy"]),
            "display_value": f"{portfolio['recommended_rollout_budget_cap_proxy']:,.0f}",
            "readout": "Budget frontier selected from ROI simulation.",
        },
    ]

    return pd.DataFrame(rows)


def make_treatment_control_balance_diagnostics(experiment_population):
    balance_features = [
        "predicted_churn_probability",
        "save_worthiness_score",
        "profit_adjusted_clv_proxy",
        "gross_value_at_risk_proxy",
        "expected_intervention_response_rate",
        "intervention_cost_proxy",
        "net_save_value_proxy",
        "actual_future_churn_label",
    ]

    rows = []

    for feature in balance_features:
        treatment_values = experiment_population.loc[
            experiment_population["experiment_group"].eq("Treatment - retention offer"),
            feature,
        ]

        control_values = experiment_population.loc[
            experiment_population["experiment_group"].eq("Control - holdout"),
            feature,
        ]

        treatment_mean = float(treatment_values.mean())
        control_mean = float(control_values.mean())
        absolute_difference = treatment_mean - control_mean

        pooled_std = float(
            np.sqrt(
                (
                    treatment_values.var(ddof=1)
                    + control_values.var(ddof=1)
                )
                / 2
            )
        )

        # Standardized differences make balance easier to compare across
        # features with very different scales.
        standardized_difference = (
            absolute_difference / pooled_std
            if pooled_std > 0
            else 0
        )

        rows.append(
            {
                "balance_feature": feature,
                "treatment_mean": treatment_mean,
                "control_mean": control_mean,
                "absolute_difference": absolute_difference,
                "standardized_difference": standardized_difference,
                "balance_status": (
                    "Balanced"
                    if abs(standardized_difference) <= 0.10
                    else "Review imbalance"
                ),
                "readout": (
                    "Treatment/control groups are balanced on this feature."
                    if abs(standardized_difference) <= 0.10
                    else "Review this feature before interpreting test results."
                ),
            }
        )

    return pd.DataFrame(rows)


def make_minimum_detectable_effect_readout(experiment_population):
    treatment_n = int(
        experiment_population["experiment_group"].eq("Treatment - retention offer").sum()
    )
    control_n = int(
        experiment_population["experiment_group"].eq("Control - holdout").sum()
    )

    n_per_group = min(treatment_n, control_n)

    baseline_churn = float(
        experiment_population.loc[
            experiment_population["experiment_group"].eq("Control - holdout"),
            "predicted_churn_probability",
        ].mean()
    )

    rows = []

    for relative_reduction in [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]:
        treatment_churn = baseline_churn * (1 - relative_reduction)
        absolute_reduction = baseline_churn - treatment_churn

        power = approximate_power_two_proportion(
            p_control=baseline_churn,
            p_treatment=treatment_churn,
            n_per_group=n_per_group,
            alpha=PRIMARY_ALPHA,
        )

        required_per_group_80 = two_proportion_sample_size(
            p_control=baseline_churn,
            p_treatment=treatment_churn,
            alpha=PRIMARY_ALPHA,
            power=0.80,
        )

        rows.append(
            {
                "relative_churn_reduction_tested": relative_reduction,
                "absolute_churn_reduction_tested": absolute_reduction,
                "baseline_churn_rate": baseline_churn,
                "treatment_churn_rate_tested": treatment_churn,
                "planned_sample_size_per_group": n_per_group,
                "approx_power_with_planned_sample": power,
                "required_sample_size_per_group_80_power": (
                    None if math.isinf(required_per_group_80) else int(required_per_group_80)
                ),
                "mde_readout": (
                    "Detectable with planned sample"
                    if power >= 0.80
                    else "Likely underpowered at this effect size"
                ),
            }
        )

    return pd.DataFrame(rows)


def make_experiment_risk_register():
    rows = [
        {
            "risk_area": "Response-rate uncertainty",
            "risk_description": "Planning response rates may overstate actual customer behavior.",
            "mitigation": "Use conservative response scenarios and require positive observed net value before rollout.",
            "owner": "Analytics + lifecycle marketing",
        },
        {
            "risk_area": "Offer cannibalization",
            "risk_description": "Some customers may accept discounts even though they would have stayed.",
            "mitigation": "Use a holdout group and track incremental saved CLV, not just redemption.",
            "owner": "Marketing strategy",
        },
        {
            "risk_area": "Customer experience",
            "risk_description": "Retention offers can create complaints, refund pressure, or perceived unfairness.",
            "mitigation": "Track support contacts, complaints, refund requests, and opt-out behavior as guardrails.",
            "owner": "Customer operations",
        },
        {
            "risk_area": "Implementation leakage",
            "risk_description": "Treatment rules may accidentally reach control customers or suppress eligible customers.",
            "mitigation": "Use deterministic customer-level assignment and preserve the assignment file.",
            "owner": "Data engineering",
        },
        {
            "risk_area": "Segment imbalance",
            "risk_description": "Treatment and control groups may differ across high-value or high-risk customer segments.",
            "mitigation": "Use stratified randomization and review balance diagnostics before launch.",
            "owner": "Analytics",
        },
        {
            "risk_area": "Short-term measurement bias",
            "risk_description": "A short readout window may miss delayed churn or delayed response.",
            "mitigation": "Report early readout separately from final readout and keep lifecycle windows consistent.",
            "owner": "Experiment owner",
        },
    ]

    return pd.DataFrame(rows)



def validate_outputs(experiment_population, outputs):
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

    design = outputs["01_experiment_design_summary"].iloc[0]
    power = outputs["02_power_sample_size_scenarios"]
    treatment_plan = outputs["03_treatment_control_plan"]
    metrics = outputs["04_metric_definitions"]
    balance = outputs["09_treatment_control_balance_diagnostics"]
    mde = outputs["10_minimum_detectable_effect_readout"]
    risk_register = outputs["11_experiment_risk_register"]

    treatment_customers = int(design["treatment_customers"])
    control_customers = int(design["control_customers"])
    total = treatment_customers + control_customers
    treatment_share = treatment_customers / total if total > 0 else 0

    add_check(
        "experiment_population_not_empty",
        len(experiment_population),
        len(experiment_population) > 0,
        "A/B candidate population should not be empty.",
    )

    add_check(
        "one_row_per_customer",
        int(experiment_population["msno"].duplicated().sum()),
        experiment_population["msno"].duplicated().sum() == 0,
        "Each customer should appear once in the A/B plan.",
    )

    add_check(
        "treatment_and_control_exist",
        f"treatment={treatment_customers}, control={control_customers}",
        treatment_customers > 0 and control_customers > 0,
        "Both treatment and control groups are required.",
    )

    add_check(
        "randomization_balance_reasonable",
        round(treatment_share, 4),
        0.45 <= treatment_share <= 0.55,
        "Treatment share should be close to 50%.",
    )

    add_check(
        "expected_test_net_value_positive",
        round(float(design["treatment_expected_net_save_value_proxy"]), 2),
        float(design["treatment_expected_net_save_value_proxy"]) > 0,
        "Treatment arm should have positive expected net save value.",
    )

    add_check(
        "power_scenarios_created",
        len(power),
        len(power) > 0,
        "Power and sample-size scenarios should exist.",
    )

    add_check(
        "metric_definitions_created",
        len(metrics),
        len(metrics) >= 4,
        "Metric definitions should include primary, business, and guardrail metrics.",
    )

    add_check(
        "treatment_plan_groups_sum_to_population",
        int(treatment_plan["customers"].sum()),
        int(treatment_plan["customers"].sum()) == len(experiment_population),
        "Treatment/control summary should match experiment population.",
    )

    max_abs_standardized_difference = float(balance["standardized_difference"].abs().max())

    add_check(
        "balance_diagnostics_created",
        len(balance),
        len(balance) > 0,
        "Treatment/control balance diagnostics should exist.",
    )

    add_check(
        "max_standardized_difference_reasonable",
        round(max_abs_standardized_difference, 6),
        max_abs_standardized_difference <= 0.10,
        "Treatment/control groups should be balanced on key planning features.",
    )

    add_check(
        "mde_readout_created",
        len(mde),
        len(mde) > 0,
        "Minimum detectable effect readout should exist.",
    )

    add_check(
        "risk_register_created",
        len(risk_register),
        len(risk_register) >= 5,
        "Experiment risk register should include practical launch risks.",
    )

    report = pd.DataFrame(checks)
    report.to_csv(OUTPUT_DIR / "_ab_test_planner_validation_report.csv", index=False)

    with open(OUTPUT_DIR / "_ab_test_planner_validation_report.json", "w") as f:
        json.dump(clean_for_json(checks), f, indent=2)

    print("\nValidation report:")
    print(report.to_string(index=False))

    failed = report[report["status"] == "FAIL"]
    if len(failed) > 0:
        raise ValueError("A/B test planner validation failed. Review outputs before moving forward.")

    return report


def write_sql_reference():
    sql_path = SQL_DIR / "10_ab_test_planner.sql"

    lines = [
        "-- 10_ab_test_planner.sql",
        "-- SQL reference for A/B test planning outputs.",
        "",
        "-- Treatment/control balance",
        "SELECT",
        "    experiment_group,",
        "    customers,",
        "    avg_predicted_churn_probability,",
        "    actual_future_churn_rate_retrospective,",
        "    gross_value_at_risk_proxy,",
        "    planned_campaign_cost_proxy",
        "FROM read_csv_auto('data/processed/ab_test_outputs/03_treatment_control_plan.csv')",
        "ORDER BY experiment_group;",
        "",
        "-- Experiment design summary",
        "SELECT",
        "    experiment_name,",
        "    candidate_customers,",
        "    treatment_customers,",
        "    control_customers,",
        "    control_churn_rate_planning,",
        "    treatment_churn_rate_planning,",
        "    expected_absolute_churn_reduction,",
        "    treatment_expected_net_save_value_proxy,",
        "    treatment_roi_proxy,",
        "    approx_power_for_base_effect",
        "FROM read_csv_auto('data/processed/ab_test_outputs/01_experiment_design_summary.csv');",
        "",
        "-- Power scenarios",
        "SELECT",
        "    scenario_label,",
        "    target_power,",
        "    required_sample_size_per_group,",
        "    planned_sample_size_per_group,",
        "    approx_power_with_planned_sample,",
        "    power_readout",
        "FROM read_csv_auto('data/processed/ab_test_outputs/02_power_sample_size_scenarios.csv')",
        "ORDER BY scenario_label, target_power;",
    ]

    with open(sql_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved SQL reference file: {sql_path}")


def write_executive_summary(outputs, validation_report):
    portfolio = outputs["00_ab_test_portfolio_summary"].iloc[0]
    design = outputs["01_experiment_design_summary"].iloc[0]

    power = outputs["02_power_sample_size_scenarios"]
    base_power = power[
        (power["scenario_label"] == "Base expected response")
        & (power["target_power"] == 0.80)
    ]

    if base_power.empty:
        base_power_row = power.iloc[0]
    else:
        base_power_row = base_power.iloc[0]

    executive = {
        "candidate_customers": int(design["candidate_customers"]),
        "treatment_customers": int(design["treatment_customers"]),
        "control_customers": int(design["control_customers"]),
        "control_churn_rate_planning": float(design["control_churn_rate_planning"]),
        "treatment_churn_rate_planning": float(design["treatment_churn_rate_planning"]),
        "expected_absolute_churn_reduction": float(design["expected_absolute_churn_reduction"]),
        "expected_relative_churn_reduction": float(design["expected_relative_churn_reduction"]),
        "expected_customers_saved_in_treatment": float(design["expected_customers_saved_in_treatment"]),
        "treatment_expected_net_save_value_proxy": float(design["treatment_expected_net_save_value_proxy"]),
        "treatment_campaign_cost_proxy": float(design["treatment_campaign_cost_proxy"]),
        "treatment_roi_proxy": float(design["treatment_roi_proxy"]),
        "break_even_response_rate": float(design["break_even_response_rate"]),
        "approx_power_for_base_effect": float(design["approx_power_for_base_effect"]),
        "required_sample_size_per_group_base_80_power": int(base_power_row["required_sample_size_per_group"]),
        "planned_sample_size_per_group": int(base_power_row["planned_sample_size_per_group"]),
        "recommended_rollout_budget_cap_proxy": float(portfolio["recommended_rollout_budget_cap_proxy"]),
        "efficient_budget_value_capture_vs_max": float(portfolio["efficient_budget_value_capture_vs_max"]),
        "validation_status": "PASS" if (validation_report["status"] == "PASS").all() else "FAIL",
    }

    with open(OUTPUT_DIR / "_ab_test_planner_summary.json", "w") as f:
        json.dump(clean_for_json(executive), f, indent=2)

    lines = [
        "# A/B Test Planner Summary",
        "",
        "Recommended experiment: Save-worthy retention offer A/B test",
        "",
        f"Candidate customers: {executive['candidate_customers']:,}",
        f"Treatment customers: {executive['treatment_customers']:,}",
        f"Control customers: {executive['control_customers']:,}",
        f"Recommended rollout budget cap proxy: {executive['recommended_rollout_budget_cap_proxy']:,.0f}",
        f"Efficient budget value capture vs max tested net value: {executive['efficient_budget_value_capture_vs_max']:.2%}",
        "",
        "Expected test impact:",
        f"- Planning control churn rate: {executive['control_churn_rate_planning']:.2%}",
        f"- Planning treatment churn rate: {executive['treatment_churn_rate_planning']:.2%}",
        f"- Expected absolute churn reduction: {executive['expected_absolute_churn_reduction']:.2%}",
        f"- Expected relative churn reduction: {executive['expected_relative_churn_reduction']:.2%}",
        f"- Expected customers saved in treatment arm: {executive['expected_customers_saved_in_treatment']:,.0f}",
        f"- Treatment expected net save value proxy: {executive['treatment_expected_net_save_value_proxy']:,.0f}",
        f"- Treatment campaign cost proxy: {executive['treatment_campaign_cost_proxy']:,.0f}",
        f"- Treatment ROI proxy: {executive['treatment_roi_proxy']:.2f}x",
        "",
        "Power readout:",
        f"- Approximate power for base expected effect: {executive['approx_power_for_base_effect']:.2%}",
        f"- Required sample per group for 80% power under base response: {executive['required_sample_size_per_group_base_80_power']:,}",
        f"- Planned sample per group: {executive['planned_sample_size_per_group']:,}",
        "",
        "Break-even readout:",
        f"- Break-even response rate: {executive['break_even_response_rate']:.2%}",
        "",
        "Decision rules:",
        "- Scale only if treatment reduces churn, creates positive incremental net value, and passes customer-experience guardrails.",
        "- Do not scale from churn reduction alone.",
        "- Kill or redesign the campaign if economics fail, even if the model ranking looks strong.",
        "",
        "Human launch-readiness readout:",
        "- The test uses deterministic stratified randomization, not a naive global split.",
        "- Balance diagnostics check whether treatment and control are comparable before launch.",
        "- Minimum detectable effect scenarios show what size of churn reduction the test can realistically detect.",
        "- The risk register documents response uncertainty, cannibalization, customer experience, implementation leakage, and timing bias.",
        "",
        "Business interpretation:",
        (
            "This final layer converts the retention engine into a testable operating plan. "
            "The project now moves from model output to accountable experimentation: who gets the offer, "
            "who is held out, what lift is required, how much value is at stake, what risks could invalidate the test, "
            "and what rules determine rollout."
        ),
    ]

    summary_path = OUTPUT_DIR / "_executive_ab_test_planner_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved {summary_path}")
    print(f"Saved {OUTPUT_DIR / '_ab_test_planner_summary.json'}")


def main():
    print("\nRunning A/B test planner...")

    #save_scores, _roi_matrix, budget_frontier, _best_strategy = load_inputs()
    save_scores, roi_matrix, budget_frontier, best_strategy = load_inputs()

    experiment_population, efficient_budget = build_experiment_population(
        save_scores=save_scores,
        budget_frontier=budget_frontier,
    )

    portfolio_summary = make_ab_test_portfolio_summary(
        save_scores=save_scores,
        experiment_population=experiment_population,
        efficient_budget=efficient_budget,
    )

    design_summary = make_experiment_design_summary(experiment_population)
    power_scenarios = make_power_sample_size_scenarios(experiment_population)
    treatment_control_plan = make_treatment_control_plan(experiment_population)
    metric_definitions = make_metric_definitions()
    randomization_plan = make_randomization_unit_plan(experiment_population)
    decision_rules = make_rollout_decision_rules(design_summary)
    segment_readiness = make_segment_test_readiness(experiment_population)
    tableau_plan = make_tableau_ab_test_plan(
        portfolio_summary=portfolio_summary,
        design_summary=design_summary,
        treatment_plan=treatment_control_plan,
        power_scenarios=power_scenarios,
    )
    balance_diagnostics = make_treatment_control_balance_diagnostics(experiment_population)
    minimum_detectable_effect = make_minimum_detectable_effect_readout(experiment_population)
    experiment_risk_register = make_experiment_risk_register()

    outputs = {
        "00_ab_test_portfolio_summary": portfolio_summary,
        "01_experiment_design_summary": design_summary,
        "02_power_sample_size_scenarios": power_scenarios,
        "03_treatment_control_plan": treatment_control_plan,
        "04_metric_definitions": metric_definitions,
        "05_randomization_unit_plan": randomization_plan,
        "06_rollout_decision_rules": decision_rules,
        "07_segment_test_readiness": segment_readiness,
        "08_tableau_ab_test_plan": tableau_plan,
        "09_treatment_control_balance_diagnostics": balance_diagnostics,
        "10_minimum_detectable_effect_readout": minimum_detectable_effect,
        "11_experiment_risk_register": experiment_risk_register,
    }

    write_sql_reference()

    experiment_population.to_parquet(
        OUTPUT_DIR / "ab_test_candidate_population.parquet",
        index=False,
    )

    experiment_population.to_csv(
        OUTPUT_DIR / "ab_test_candidate_population.csv",
        index=False,
    )

    experiment_population.head(100000).to_csv(
        OUTPUT_DIR / "tableau_ab_test_customer_sample.csv",
        index=False,
    )

    print(f"Saved {OUTPUT_DIR / 'ab_test_candidate_population.parquet'}")
    print(f"Saved {OUTPUT_DIR / 'ab_test_candidate_population.csv'}")
    print(f"Saved {OUTPUT_DIR / 'tableau_ab_test_customer_sample.csv'}")

    for name, df in outputs.items():
        path = OUTPUT_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {path} | rows={len(df):,} cols={df.shape[1]:,}")

    with open(OUTPUT_DIR / "_ab_test_planner_assumptions.json", "w") as f:
        json.dump(
            clean_for_json(
                {
                    "randomization_seed": RANDOMIZATION_SEED,
                    "primary_alpha": PRIMARY_ALPHA,
                    "default_treatment_share": DEFAULT_TREATMENT_SHARE,
                    "response_multiplier_scenarios": RESPONSE_MULTIPLIER_SCENARIOS,
                    "note": "This script designs the experiment. It does not claim the campaign has already worked.",
                }
            ),
            f,
            indent=2,
        )

    validation_report = validate_outputs(experiment_population, outputs)
    write_executive_summary(outputs, validation_report)

    print("\n10_ab_test_planner.py complete.")


if __name__ == "__main__":
    main()
