"""
11_build_tableau_exports.py

Tableau export layer for the Customer Revenue Recovery & Retention ROI Engine.

The modeling scripts create many detailed outputs. Tableau should not have to
rebuild all of that business logic with messy calculated fields.

This script creates clean, dashboard-ready CSVs for:
1. Executive KPI cards
2. Value leakage heatmaps
3. Risk-value decision maps
4. Save-worthiness funnel views
5. ROI budget frontier charts
6. A/B test readiness cards
7. Model governance summaries
8. Executive recommendation cards

The goal is to keep Tableau focused on presentation while Python owns the
analytics, assumptions, and business logic.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "tableau_exports"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(rel_path: str) -> pd.DataFrame:
    path = PROCESSED / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, filename: str) -> None:
    path = OUT / filename
    df.to_csv(path, index=False)
    print(f"Created {path.relative_to(ROOT)} | {len(df):,} rows x {len(df.columns):,} columns")


# Display helpers keep Tableau labels consistent across KPI cards, tooltips,
# and executive recommendation views.
def money(x, decimals=2):
    if pd.isna(x):
        return ""
    x = float(x)
    if abs(x) >= 1_000_000_000:
        return f"${x/1_000_000_000:.{decimals}f}B"
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.{decimals}f}M"
    if abs(x) >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:,.0f}"


def pct(x, decimals=2):
    if pd.isna(x):
        return ""
    return f"{float(x) * 100:.{decimals}f}%"


def num(x):
    if pd.isna(x):
        return ""
    return f"{float(x):,.0f}"


# Load the final outputs from each analytical layer.
# This script should not recompute model logic; it only reshapes validated outputs
# into Tableau-friendly tables.
clv_deciles = read_csv("revenue_leakage_outputs/04_clv_concentration_deciles.csv")
leakage_base = read_csv("revenue_leakage_outputs/08_tableau_revenue_leakage_base.csv")

risk_deciles = read_csv("churn_model_outputs/01_risk_decile_business_summary.csv")
sensitivity = read_csv("model_validation_outputs/07_cancellation_signal_sensitivity_readout.csv")

lifecycle_quadrants = read_csv("survival_lifecycle_outputs/03_value_risk_quadrant_summary.csv")
lifecycle_segments = read_csv("survival_lifecycle_outputs/04_model_priority_lifecycle_segments.csv")

save_portfolio = read_csv("save_worthiness_outputs/00_save_worthiness_portfolio_summary.csv").iloc[0]
retention_actions = read_csv("save_worthiness_outputs/01_retention_action_summary.csv")

budget_frontier = read_csv("retention_roi_outputs/03_budget_cap_frontier.csv")

ab_portfolio = read_csv("ab_test_outputs/00_ab_test_portfolio_summary.csv").iloc[0]
experiment = read_csv("ab_test_outputs/01_experiment_design_summary.csv").iloc[0]


# Locked executive headline values from the final project summary.
# These are kept explicit so the Tableau board, README, and project narrative
# use the same headline figures.
ANNUAL_REVENUE_RUN_RATE_PROXY = 1_690_000_000
ANNUAL_MARGIN_RUN_RATE_PROXY = 1_100_000_000
TOP_5_FUTURE_CHURNED_CLV_SHARE = 0.7809
TOP_RECOVERY_FUTURE_CHURNED_REVENUE_PROXY = 10_893_857


# Helpful derived values
total_customers = float(save_portfolio["customers"])
total_clv = float(save_portfolio["profit_adjusted_clv_proxy"])
future_churned_clv = float(save_portfolio["future_churned_clv_proxy"])
if total_customers <= 0 or total_clv <= 0:
    raise ValueError("Executive export requires positive customer count and positive total CLV.")

clv_leakage_rate = future_churned_clv / total_clv

high_risk_high_value = lifecycle_quadrants.loc[
    lifecycle_quadrants["value_risk_quadrant"].astype(str).str.contains("High risk / high value", na=False)
]
high_risk_high_value_customers = high_risk_high_value["customers"].sum()
high_risk_high_value_future_clv = high_risk_high_value["future_churned_clv_proxy"].sum()

top_clv_decile = clv_deciles.sort_values("clv_rank_decile").iloc[0]
top_risk_decile = risk_deciles.sort_values("risk_decile").iloc[0]
sensitivity_row = sensitivity.iloc[0]


# 01. Executive KPI summary
# One row per KPI keeps the Tableau executive card layout simple:
# section, metric, raw value, display value, readout, and sort order.
kpi_rows = []


def add_kpi(section, metric, value, display_value, readout, sort_order):
    kpi_rows.append(
        {
            "section": section,
            "metric": metric,
            "value": value,
            "display_value": display_value,
            "readout": readout,
            "sort_order": sort_order,
        }
    )


add_kpi("Portfolio scale", "Customers scored", total_customers, num(total_customers), "Full customer population scored by the retention engine.", 1)
add_kpi("Portfolio scale", "Annual revenue run-rate proxy", ANNUAL_REVENUE_RUN_RATE_PROXY, money(ANNUAL_REVENUE_RUN_RATE_PROXY), "Portfolio-scale annual revenue baseline used for executive sizing.", 2)
add_kpi("Portfolio scale", "Annual margin run-rate proxy", ANNUAL_MARGIN_RUN_RATE_PROXY, money(ANNUAL_MARGIN_RUN_RATE_PROXY), "Margin-adjusted annual portfolio baseline.", 3)
add_kpi("Portfolio scale", "Profit-adjusted CLV proxy", total_clv, money(total_clv), "Estimated customer value after margin assumptions.", 4)
add_kpi("Portfolio leakage", "Future churned CLV proxy", future_churned_clv, money(future_churned_clv), "Estimated customer value exposed to future churn.", 5)
add_kpi("Portfolio leakage", "CLV leakage rate", clv_leakage_rate, pct(clv_leakage_rate), "Share of profit-adjusted CLV exposed to future churn.", 6)

add_kpi("Value concentration", "Top 10% CLV share", float(top_clv_decile["clv_share"]), pct(top_clv_decile["clv_share"]), "Top CLV decile share of total profit-adjusted CLV.", 7)
add_kpi("Value concentration", "Top 10% future churned CLV share", float(top_clv_decile["future_churned_clv_share"]), pct(top_clv_decile["future_churned_clv_share"]), "Top CLV decile share of future churned CLV proxy.", 8)
add_kpi("Value concentration", "Top 5% future churned CLV share", TOP_5_FUTURE_CHURNED_CLV_SHARE, pct(TOP_5_FUTURE_CHURNED_CLV_SHARE), "Top 5% of customers account for most future churned CLV proxy.", 9)

add_kpi("Recovery segment", "Top recovery segment customers", 24676, num(24676), "Long-tenure / high engagement / high revenue segment.", 10)
add_kpi("Recovery segment", "Top recovery segment churn risk", 0.4339, pct(0.4339), "Future churn risk in top recovery segment.", 11)
add_kpi("Recovery segment", "Top recovery segment future churned revenue proxy", TOP_RECOVERY_FUTURE_CHURNED_REVENUE_PROXY, money(TOP_RECOVERY_FUTURE_CHURNED_REVENUE_PROXY), "Future churned revenue proxy in the strongest early recovery segment.", 12)

add_kpi("Save-worthiness", "Paid-offer candidates", float(save_portfolio["paid_offer_customers"]), num(save_portfolio["paid_offer_customers"]), "Customers clearing positive paid-offer economics.", 13)
add_kpi("Save-worthiness", "Paid-offer candidate rate", float(save_portfolio["paid_offer_rate"]), pct(save_portfolio["paid_offer_rate"]), "Share of total customers selected for paid-offer candidacy.", 14)
add_kpi("Save-worthiness", "Expected saved CLV proxy", float(save_portfolio["paid_offer_expected_saved_clv_proxy"]), money(save_portfolio["paid_offer_expected_saved_clv_proxy"]), "Expected saved value under base planning assumptions.", 15)
add_kpi("Save-worthiness", "Estimated intervention cost proxy", float(save_portfolio["paid_offer_intervention_cost_proxy"]), money(save_portfolio["paid_offer_intervention_cost_proxy"]), "Estimated paid-offer campaign cost proxy.", 16)
add_kpi("Save-worthiness", "Expected net save-value proxy", float(save_portfolio["paid_offer_net_save_value_proxy"]), money(save_portfolio["paid_offer_net_save_value_proxy"]), "Expected net value after intervention cost under planning assumptions.", 17)
add_kpi("Save-worthiness", "Paid-offer ROI proxy", float(save_portfolio["paid_offer_roi_proxy"]), f"{float(save_portfolio['paid_offer_roi_proxy']):.2f}x", "Planning ROI proxy for paid-offer candidates.", 18)

add_kpi("ROI strategy", "Efficient budget frontier", 750000, "$750K", "Budget point that captures nearly all tested net value with less spend.", 19)
add_kpi("ROI strategy", "Value capture vs max tested net value", float(ab_portfolio["efficient_budget_value_capture_vs_max"]), pct(ab_portfolio["efficient_budget_value_capture_vs_max"]), "Share of maximum tested net value captured at the efficient budget cap.", 20)
add_kpi("A/B test", "A/B candidate customers", float(ab_portfolio["ab_test_candidate_customers"]), num(ab_portfolio["ab_test_candidate_customers"]), "Efficient-frontier population selected for experiment planning.", 21)
add_kpi("A/B test", "Treatment customers", float(ab_portfolio["treatment_customers"]), num(ab_portfolio["treatment_customers"]), "Customers assigned to treatment arm.", 22)
add_kpi("A/B test", "Control customers", float(ab_portfolio["control_customers"]), num(ab_portfolio["control_customers"]), "Holdout customers for causal measurement.", 23)

write_csv(pd.DataFrame(kpi_rows), "01_executive_kpi_summary.csv")


# 02. Value leakage heatmap
# Pre-aggregating the risk/value grid in Python prevents Tableau from needing
# complex joins or repeated calculated fields.
heatmap = (
    lifecycle_segments
    .groupby(["churn_risk_tier", "clv_value_tier"], dropna=False)
    .agg(
        customers=("customers", "sum"),
        avg_predicted_churn_probability=("avg_predicted_churn_probability", "mean"),
        future_churn_rate=("future_churn_rate", "mean"),
        profit_adjusted_clv_proxy=("profit_adjusted_clv_proxy", "sum"),
        future_churned_clv_proxy=("future_churned_clv_proxy", "sum"),
        avg_lifecycle_duration_months=("avg_lifecycle_duration_months", "mean"),
        avg_monthly_value_baseline=("avg_monthly_value_baseline", "mean"),
    )
    .reset_index()
)
heatmap["avg_clv_per_customer"] = heatmap["profit_adjusted_clv_proxy"] / heatmap["customers"].replace(0, np.nan)
heatmap["future_churned_clv_display"] = heatmap["future_churned_clv_proxy"].apply(money)
heatmap["customer_display"] = heatmap["customers"].apply(num)
heatmap["heatmap_label"] = heatmap["future_churned_clv_display"] + "\n" + heatmap["customer_display"] + " customers"
write_csv(heatmap, "02_value_leakage_heatmap.csv")


# 03. Risk-value decision map
# This view supports the executive question: where is churn risk commercially meaningful?
decision_map = lifecycle_segments.copy()
decision_map["segment_label"] = (
    decision_map["lifecycle_stage"].astype(str)
    + " / "
    + decision_map["engagement_tier"].astype(str)
    + " / "
    + decision_map["revenue_tier"].astype(str)
    + " / "
    + decision_map["clv_value_tier"].astype(str)
)
decision_map["avg_clv_per_customer"] = decision_map["profit_adjusted_clv_proxy"] / decision_map["customers"].replace(0, np.nan)
decision_map["bubble_value"] = decision_map["future_churned_clv_proxy"]
decision_map["decision_zone"] = np.select(
    [
        decision_map["value_risk_quadrant"].astype(str).str.contains("High risk / high value", na=False),
        decision_map["churn_risk_tier"].astype(str).str.contains("Critical|High", case=False, na=False),
        decision_map["clv_value_tier"].astype(str).str.contains("Elite|High", case=False, na=False),
    ],
    [
        "Save-worthiness review",
        "Risk monitor",
        "Value monitor",
    ],
    default="No paid action focus",
)
decision_map = decision_map.sort_values("lifecycle_model_priority_score", ascending=False)
write_csv(decision_map, "03_risk_value_decision_map.csv")


# 04. Save-worthiness funnel
# The funnel shows how the project narrows from all customers to the testable
# treatment/control population.
funnel_rows = [
    {
        "stage_order": 1,
        "stage": "Customers scored",
        "customers": total_customers,
        "display_value": num(total_customers),
        "stage_type": "Population",
        "readout": "Full customer population scored by the retention engine.",
    },
    {
        "stage_order": 2,
        "stage": "High-risk / high-value customers",
        "customers": high_risk_high_value_customers,
        "display_value": num(high_risk_high_value_customers),
        "stage_type": "Risk-value filter",
        "readout": f"{money(high_risk_high_value_future_clv)} future churned CLV proxy in high-risk / high-value customers.",
    },
    {
        "stage_order": 3,
        "stage": "Economically positive paid-offer candidates",
        "customers": float(save_portfolio["paid_offer_customers"]),
        "display_value": num(save_portfolio["paid_offer_customers"]),
        "stage_type": "Save-worthiness filter",
        "readout": f"{money(save_portfolio['paid_offer_net_save_value_proxy'])} expected net save-value proxy under planning assumptions.",
    },
    {
        "stage_order": 4,
        "stage": "A/B test candidate customers",
        "customers": float(ab_portfolio["ab_test_candidate_customers"]),
        "display_value": num(ab_portfolio["ab_test_candidate_customers"]),
        "stage_type": "Experiment population",
        "readout": "Efficient-frontier population selected for controlled test planning.",
    },
    {
        "stage_order": 5,
        "stage": "Treatment customers",
        "customers": float(ab_portfolio["treatment_customers"]),
        "display_value": num(ab_portfolio["treatment_customers"]),
        "stage_type": "A/B split",
        "readout": "Customers assigned to receive retention intervention.",
    },
    {
        "stage_order": 6,
        "stage": "Control customers",
        "customers": float(ab_portfolio["control_customers"]),
        "display_value": num(ab_portfolio["control_customers"]),
        "stage_type": "A/B split",
        "readout": "Holdout customers used to measure incremental impact.",
    },
]
funnel = pd.DataFrame(funnel_rows)
funnel["share_of_total_customers"] = funnel["customers"] / total_customers
write_csv(funnel, "04_save_worthiness_funnel.csv")


# 05. ROI budget frontier
# This export powers the capital allocation view: which budget captures most
# of the tested net value without unnecessary spend?
frontier = budget_frontier.copy()
max_net_value = frontier["net_save_value_proxy"].max()
frontier["value_capture_vs_max"] = frontier["net_save_value_proxy"] / max_net_value
frontier["budget_label"] = frontier["budget_cap"].apply(money)
frontier["net_value_display"] = frontier["net_save_value_proxy"].apply(money)
frontier["campaign_cost_display"] = frontier["campaign_cost_proxy"].apply(money)
frontier["roi_display"] = frontier["roi_proxy"].map(lambda x: f"{x:.2f}x")
frontier["efficient_frontier_flag"] = np.where(frontier["budget_cap"].eq(750000), "Efficient frontier", "Other tested budget")
frontier["max_net_value_flag"] = np.where(frontier["net_save_value_proxy"].eq(max_net_value), "Max tested net value", "Other tested budget")
frontier["frontier_readout"] = np.where(
    frontier["budget_cap"].eq(750000),
    "Recommended efficient frontier: captures nearly all tested net value with less committed budget.",
    "Budget scenario tested by ROI simulator."
)
write_csv(frontier, "05_roi_budget_frontier.csv")


# 06. A/B test readiness
# These rows become executive cards explaining why the campaign should be tested
# before any full rollout.
ab_rows = [
    ("Experiment overview", "Candidate customers", experiment["candidate_customers"], num(experiment["candidate_customers"]), "Efficient-frontier save-worthy population selected for testing.", 1),
    ("Experiment overview", "Treatment customers", experiment["treatment_customers"], num(experiment["treatment_customers"]), "Customers assigned to receive the retention intervention.", 2),
    ("Experiment overview", "Control customers", experiment["control_customers"], num(experiment["control_customers"]), "Holdout customers for causal measurement.", 3),
    ("Experiment economics", "Treatment expected saved CLV proxy", experiment["treatment_expected_saved_clv_proxy"], money(experiment["treatment_expected_saved_clv_proxy"]), "Expected saved CLV in the treatment arm under planning assumptions.", 4),
    ("Experiment economics", "Treatment campaign cost proxy", experiment["treatment_campaign_cost_proxy"], money(experiment["treatment_campaign_cost_proxy"]), "Estimated treatment campaign cost proxy.", 5),
    ("Experiment economics", "Treatment expected net save-value proxy", experiment["treatment_expected_net_save_value_proxy"], money(experiment["treatment_expected_net_save_value_proxy"]), "Expected treatment net value after campaign cost.", 6),
    ("Experiment economics", "Treatment ROI proxy", experiment["treatment_roi_proxy"], f"{float(experiment['treatment_roi_proxy']):.2f}x", "Planning ROI proxy for treatment arm.", 7),
    ("Experiment economics", "Break-even response rate", experiment["break_even_response_rate"], pct(experiment["break_even_response_rate"]), "Minimum response rate needed to clear campaign cost.", 8),
    ("Experiment power", "Approximate power for base effect", experiment["approx_power_for_base_effect"], pct(experiment["approx_power_for_base_effect"]), "Planning power for the base expected effect.", 9),
    ("Experiment effect", "Expected absolute churn reduction", experiment["expected_absolute_churn_reduction"], pct(experiment["expected_absolute_churn_reduction"]), "Planned treatment-control churn difference.", 10),
    ("Experiment effect", "Expected relative churn reduction", experiment["expected_relative_churn_reduction"], pct(experiment["expected_relative_churn_reduction"]), "Expected relative churn reduction under planning assumptions.", 11),
    ("Decision rule", "Scale rule", np.nan, "Net value + churn + guardrails", "Do not scale from churn reduction alone. Scale only if churn decreases, incremental net value is positive, and customer-experience guardrails pass.", 12),
]
ab_ready = pd.DataFrame(
    ab_rows,
    columns=["section", "metric", "value", "display_value", "readout", "sort_order"]
)
write_csv(ab_ready, "06_ab_test_readiness.csv")


# 07. Model governance summary
# This keeps model quality and leakage/sensitivity checks visible inside the dashboard,
# not buried in the code.
model_rows = [
    ("Main churn model", "Model", np.nan, "HistGradientBoosting", "Best-performing churn-risk model used for business ranking.", 1),
    ("Main churn model", "ROC-AUC", sensitivity_row["main_model_roc_auc"], f"{float(sensitivity_row['main_model_roc_auc']):.4f}", "Main model discrimination performance.", 2),
    ("Main churn model", "PR-AUC", sensitivity_row["main_model_pr_auc"], f"{float(sensitivity_row['main_model_pr_auc']):.4f}", "Main model precision-recall performance.", 3),
    ("Main churn model", "Top-decile lift", top_risk_decile["lift_vs_portfolio"], f"{float(top_risk_decile['lift_vs_portfolio']):.2f}x", "Top risk decile lift versus portfolio baseline.", 4),
    ("Main churn model", "Top-decile churner capture", top_risk_decile["churn_capture_share"], pct(top_risk_decile["churn_capture_share"]), "Share of churners captured by highest-risk decile.", 5),
    ("Main churn model", "Top-decile future churned CLV capture", top_risk_decile["future_churned_clv_share"], pct(top_risk_decile["future_churned_clv_share"]), "Share of future churned CLV proxy captured by highest-risk decile.", 6),
    ("No-cancellation-signal sensitivity model", "ROC-AUC", sensitivity_row["no_cancel_model_roc_auc"], f"{float(sensitivity_row['no_cancel_model_roc_auc']):.4f}", "Sensitivity model performance after removing direct cancellation signals.", 7),
    ("No-cancellation-signal sensitivity model", "PR-AUC", sensitivity_row["no_cancel_model_pr_auc"], f"{float(sensitivity_row['no_cancel_model_pr_auc']):.4f}", "Sensitivity model precision-recall performance after removing direct cancellation signals.", 8),
    ("No-cancellation-signal sensitivity model", "Top-decile lift", sensitivity_row["no_cancel_top_decile_lift"], f"{float(sensitivity_row['no_cancel_top_decile_lift']):.2f}x", "Risk ranking remained strong without obvious cancellation indicators.", 9),
    ("Governance", "Cancellation-signal sensitivity", np.nan, str(sensitivity_row["status"]), str(sensitivity_row["readout"]), 10),
    ("Governance", "Feature importance review", np.nan, "Included", "Feature-importance governance included before business recommendation.", 11),
    ("Governance", "Leakage checks", np.nan, "Passed", "Leakage and governance checks passed before final decision logic.", 12),
]
model_governance = pd.DataFrame(
    model_rows,
    columns=["section", "metric", "value", "display_value", "readout", "sort_order"]
)
write_csv(model_governance, "07_model_governance_summary.csv")


# 08. Lifecycle priority segments
lifecycle_priority = lifecycle_segments.sort_values(
    "lifecycle_model_priority_score", ascending=False
).copy()
lifecycle_priority["priority_rank"] = np.arange(1, len(lifecycle_priority) + 1)
lifecycle_priority["segment_label"] = (
    lifecycle_priority["lifecycle_stage"].astype(str)
    + " / "
    + lifecycle_priority["engagement_tier"].astype(str)
    + " / "
    + lifecycle_priority["revenue_tier"].astype(str)
    + " / "
    + lifecycle_priority["clv_value_tier"].astype(str)
    + " / "
    + lifecycle_priority["churn_risk_tier"].astype(str)
)
lifecycle_priority["future_churned_clv_display"] = lifecycle_priority["future_churned_clv_proxy"].apply(money)
lifecycle_priority["customer_display"] = lifecycle_priority["customers"].apply(num)
write_csv(lifecycle_priority, "08_lifecycle_priority_segments.csv")


# 09. Executive recommendation cards
# These rows translate the analysis into a decision memo format inside Tableau.
recommendation_rows = [
    ("Recommended strategy", "Target save-worthy paid-offer candidates near the efficient budget frontier.", "Strategy", 1),
    ("Target population", f"{num(save_portfolio['paid_offer_customers'])} economically positive paid-offer candidates.", "Target", 2),
    ("Budget recommendation", "$750K efficient frontier captures 99.99% of maximum tested net value.", "Budget", 3),
    ("Expected value", f"{money(save_portfolio['paid_offer_net_save_value_proxy'])} expected net save-value proxy under base planning assumptions.", "Economics", 4),
    ("ROI proxy", f"{float(save_portfolio['paid_offer_roi_proxy']):.2f}x paid-offer ROI proxy.", "Economics", 5),
    ("A/B test requirement", f"Test {num(experiment['candidate_customers'])} candidates with {num(experiment['treatment_customers'])} treatment and {num(experiment['control_customers'])} control customers.", "Validation", 6),
    ("Scale rule", "Do not scale from churn reduction alone. Scale only if churn falls, incremental net value is positive, and customer-experience guardrails pass.", "Decision rule", 7),
]
recommendation_cards = pd.DataFrame(
    recommendation_rows,
    columns=["card_title", "card_text", "card_type", "sort_order"]
)
write_csv(recommendation_cards, "09_executive_recommendation_cards.csv")


# 10. CLV concentration deciles
clv_concentration = clv_deciles.copy()
clv_concentration["decile_label"] = clv_concentration["clv_rank_decile"].map(lambda x: f"CLV Decile {int(x)}")
clv_concentration["clv_share_display"] = clv_concentration["clv_share"].apply(pct)
clv_concentration["future_churned_clv_share_display"] = clv_concentration["future_churned_clv_share"].apply(pct)
clv_concentration["future_churned_clv_display"] = clv_concentration["future_churned_clv_proxy"].apply(money)
clv_concentration["customer_display"] = clv_concentration["customers"].apply(num)
clv_concentration["top_decile_flag"] = np.where(clv_concentration["clv_rank_decile"].eq(1), "Top 10% by CLV", "Other CLV decile")
write_csv(clv_concentration, "10_clv_concentration_deciles.csv")


# 11. Risk decile lift and capture
risk_lift = risk_deciles.copy()
risk_lift["risk_decile_label"] = risk_lift["risk_decile"].map(lambda x: f"Risk Decile {int(x)}")
risk_lift["lift_display"] = risk_lift["lift_vs_portfolio"].map(lambda x: f"{float(x):.2f}x")
risk_lift["churn_capture_display"] = risk_lift["churn_capture_share"].apply(pct)
risk_lift["future_churned_clv_share_display"] = risk_lift["future_churned_clv_share"].apply(pct)
risk_lift["top_risk_decile_flag"] = np.where(risk_lift["risk_decile"].eq(1), "Top risk decile", "Other risk decile")
write_csv(risk_lift, "11_risk_decile_lift_capture.csv")


# 12. Retention action strategy
action_strategy = retention_actions.copy()
action_strategy["net_save_value_display"] = action_strategy["net_save_value_proxy"].apply(money)
action_strategy["expected_saved_clv_display"] = action_strategy["expected_saved_clv_proxy"].apply(money)
action_strategy["intervention_cost_display"] = action_strategy["intervention_cost_proxy"].apply(money)
action_strategy["roi_display"] = action_strategy["roi_proxy"].map(lambda x: f"{float(x):.2f}x" if pd.notna(x) else "")
action_strategy = action_strategy.sort_values("net_save_value_proxy", ascending=False)
action_strategy["strategy_rank"] = np.arange(1, len(action_strategy) + 1)
write_csv(action_strategy, "12_retention_action_strategy.csv")


print("\nTableau export layer created successfully.")
print(f"Export folder: {OUT}")
