"""
09_retention_roi_simulation.py

Retention ROI simulation layer for the Customer Revenue Recovery & Retention ROI Engine.

The save-worthiness layer decides who is worth saving under one base set of assumptions.
This script stress-tests that decision.

It asks:
1. What happens if response rates are lower than expected?
2. What happens if intervention costs are higher?
3. Which targeting strategy produces the best net value?
4. How much budget should leadership allocate?
5. Where does the campaign break even?
6. Which customer pool should move into the A/B test planner?

This is not final campaign truth. It is a planning simulator. The A/B test layer later
turns this into an experiment design.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed")
SAVE_DIR = PROCESSED_DIR / "save_worthiness_outputs"
OUTPUT_DIR = PROCESSED_DIR / "retention_roi_outputs"
SQL_DIR = Path("sql")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SQL_DIR.mkdir(parents=True, exist_ok=True)

SAVE_SCORES_PATH = SAVE_DIR / "customer_save_worthiness_scores.parquet"

REQUIRED_COLUMNS = [
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

ROI_SCENARIOS = [
    {
        "scenario_name": "Conservative",
        "response_multiplier": 0.70,
        "cost_multiplier": 1.25,
        "budget_cap": 500000,
        "scenario_readout": "Lower response and higher cost; use for downside planning.",
    },
    {
        "scenario_name": "Base",
        "response_multiplier": 1.00,
        "cost_multiplier": 1.00,
        "budget_cap": 1000000,
        "scenario_readout": "Base planning case from save-worthiness assumptions.",
    },
    {
        "scenario_name": "Upside",
        "response_multiplier": 1.25,
        "cost_multiplier": 0.90,
        "budget_cap": 1500000,
        "scenario_readout": "Higher response and lower cost; use for upside planning.",
    },
    {
        "scenario_name": "Budget constrained",
        "response_multiplier": 0.90,
        "cost_multiplier": 1.10,
        "budget_cap": 250000,
        "scenario_readout": "Smaller leadership-approved campaign budget.",
    },
    {
        "scenario_name": "Aggressive growth",
        "response_multiplier": 1.40,
        "cost_multiplier": 1.15,
        "budget_cap": 2000000,
        "scenario_readout": "Larger campaign push with higher execution cost.",
    },
]

BUDGET_CAPS = [
    100000,
    250000,
    500000,
    750000,
    1000000,
    1500000,
    2000000,
]

RESPONSE_MULTIPLIERS = [
    0.50,
    0.70,
    0.85,
    1.00,
    1.15,
    1.30,
    1.50,
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


def load_save_worthiness_scores():
    require_file(SAVE_SCORES_PATH, "src/08_save_worthiness_scoring.py")

    print("\nLoading save-worthiness scores...")
    df = pd.read_parquet(SAVE_SCORES_PATH)

    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns from save-worthiness file: {missing}")

    df = df.copy()

    numeric_cols = [
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
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["actual_future_churn_label"] = pd.to_numeric(
        df["actual_future_churn_label"],
        errors="coerce",
    ).fillna(0).astype(int)

    df["should_receive_paid_offer"] = df["should_receive_paid_offer"].astype(bool)

    print(f"Loaded customers: {len(df):,}")
    print(f"Paid-offer candidates from 08: {df['should_receive_paid_offer'].sum():,}")

    return df


def get_targeting_strategies(df):
    return [
        {
            "strategy_name": "Tier 1 executive save only",
            "strategy_type": "Priority tier",
            "filter": df["save_priority_tier"].eq("Tier 1 - Executive save priority"),
            "strategy_readout": "Smallest, highest-individual-priority executive review pool.",
        },
        {
            "strategy_name": "Tier 1 plus Tier 2 premium save",
            "strategy_type": "Priority tier",
            "filter": df["save_priority_tier"].isin(
                [
                    "Tier 1 - Executive save priority",
                    "Tier 2 - Premium save priority",
                ]
            ),
            "strategy_readout": "Premium campaign pool with high expected economic value.",
        },
        {
            "strategy_name": "Tiers 1 to 3 targeted save",
            "strategy_type": "Priority tier",
            "filter": df["save_priority_tier"].isin(
                [
                    "Tier 1 - Executive save priority",
                    "Tier 2 - Premium save priority",
                    "Tier 3 - Targeted save priority",
                ]
            ),
            "strategy_readout": "Focused paid-retention campaign population.",
        },
        {
            "strategy_name": "Tiers 1 to 4 test-and-learn",
            "strategy_type": "Priority tier",
            "filter": df["save_priority_tier"].isin(
                [
                    "Tier 1 - Executive save priority",
                    "Tier 2 - Premium save priority",
                    "Tier 3 - Targeted save priority",
                    "Tier 4 - Test-and-learn save pool",
                ]
            ),
            "strategy_readout": "Controlled campaign test population before wider rollout.",
        },
        {
            "strategy_name": "All positive ROI paid-offer candidates",
            "strategy_type": "Full eligible pool",
            "filter": df["should_receive_paid_offer"],
            "strategy_readout": "Full economically positive paid-offer population from save-worthiness scoring.",
        },
        {
            "strategy_name": "Immediate premium save offer only",
            "strategy_type": "Action",
            "filter": df["recommended_retention_action"].eq("Immediate premium save offer")
            & df["should_receive_paid_offer"],
            "strategy_readout": "Highest urgency premium-save action group.",
        },
        {
            "strategy_name": "Critical risk elite/high value",
            "strategy_type": "Risk/value",
            "filter": df["should_receive_paid_offer"]
            & df["churn_risk_tier"].eq("Critical risk")
            & df["clv_value_tier"].isin(["Elite value", "High value"]),
            "strategy_readout": "Critical churn risk with strong customer value.",
        },
        {
            "strategy_name": "Premium save budget only",
            "strategy_type": "Budget tier",
            "filter": df["should_receive_paid_offer"]
            & df["retention_budget_tier"].eq("Premium save budget"),
            "strategy_readout": "Paid-offer candidates already assigned premium save budget.",
        },
    ]


def make_top_n_strategy(df, n):
    eligible = df[df["should_receive_paid_offer"]].copy()
    eligible = eligible.sort_values("net_save_value_proxy", ascending=False)

    selected_msno = set(eligible.head(n)["msno"])

    return {
        "strategy_name": f"Top {n:,} by net save value",
        "strategy_type": "Top-N",
        "filter": df["msno"].isin(selected_msno),
        "strategy_readout": f"Top {n:,} customers ranked by expected net save value.",
    }


def calculate_roi_metrics(targeted, scenario_name, response_multiplier, cost_multiplier, budget_cap):
    if targeted.empty:
        return {
            "targeted_customers": 0,
            "avg_predicted_churn_probability": 0,
            "actual_future_churn_rate": 0,
            "profit_adjusted_clv_proxy": 0,
            "future_churned_clv_proxy": 0,
            "gross_value_at_risk_proxy": 0,
            "adjusted_expected_response_rate": 0,
            "expected_customers_saved": 0,
            "expected_saved_clv_proxy": 0,
            "campaign_cost_proxy": 0,
            "net_save_value_proxy": 0,
            "roi_proxy": np.nan,
            "break_even_response_rate": np.nan,
            "budget_cap": budget_cap,
            "budget_utilization": 0,
            "budget_status": "No targets",
            "scenario_name": scenario_name,
        }

    adjusted_response = (
        targeted["expected_intervention_response_rate"] * response_multiplier
    ).clip(lower=0, upper=0.60)

    adjusted_cost = targeted["intervention_cost_proxy"] * cost_multiplier

    gross_value_at_risk = targeted["gross_value_at_risk_proxy"].sum()
    expected_saved_clv = (targeted["gross_value_at_risk_proxy"] * adjusted_response).sum()
    campaign_cost = adjusted_cost.sum()
    net_value = expected_saved_clv - campaign_cost

    expected_customers_saved = (
        targeted["predicted_churn_probability"] * adjusted_response
    ).sum()

    roi = net_value / campaign_cost if campaign_cost > 0 else np.nan

    break_even_response_rate = (
        campaign_cost / gross_value_at_risk
        if gross_value_at_risk > 0
        else np.nan
    )

    budget_utilization = campaign_cost / budget_cap if budget_cap > 0 else np.nan

    if campaign_cost <= budget_cap and net_value > 0:
        budget_status = "Within budget and positive ROI"
    elif campaign_cost <= budget_cap and net_value <= 0:
        budget_status = "Within budget but weak economics"
    elif campaign_cost > budget_cap and net_value > 0:
        budget_status = "Positive ROI but over budget"
    else:
        budget_status = "Over budget and weak economics"

    return {
        "targeted_customers": int(len(targeted)),
        "avg_predicted_churn_probability": float(targeted["predicted_churn_probability"].mean()),
        "actual_future_churn_rate": float(targeted["actual_future_churn_label"].mean()),
        "profit_adjusted_clv_proxy": float(targeted["profit_adjusted_clv_proxy"].sum()),
        "future_churned_clv_proxy": float(targeted["future_churned_clv_proxy"].sum()),
        "gross_value_at_risk_proxy": float(gross_value_at_risk),
        "adjusted_expected_response_rate": float(adjusted_response.mean()),
        "expected_customers_saved": float(expected_customers_saved),
        "expected_saved_clv_proxy": float(expected_saved_clv),
        "campaign_cost_proxy": float(campaign_cost),
        "net_save_value_proxy": float(net_value),
        "roi_proxy": None if pd.isna(roi) else float(roi),
        "break_even_response_rate": None if pd.isna(break_even_response_rate) else float(break_even_response_rate),
        "budget_cap": float(budget_cap),
        "budget_utilization": None if pd.isna(budget_utilization) else float(budget_utilization),
        "budget_status": budget_status,
        "scenario_name": scenario_name,
    }


def simulate_strategy_scenario_matrix(df):
    strategies = get_targeting_strategies(df)

    for n in [5000, 10000, 25000, 50000]:
        strategies.append(make_top_n_strategy(df, n))

    rows = []

    for strategy in strategies:
        targeted = df[strategy["filter"]].copy()

        for scenario in ROI_SCENARIOS:
            metrics = calculate_roi_metrics(
                targeted=targeted,
                scenario_name=scenario["scenario_name"],
                response_multiplier=scenario["response_multiplier"],
                cost_multiplier=scenario["cost_multiplier"],
                budget_cap=scenario["budget_cap"],
            )

            row = {
                "strategy_name": strategy["strategy_name"],
                "strategy_type": strategy["strategy_type"],
                "strategy_readout": strategy["strategy_readout"],
                "scenario_readout": scenario["scenario_readout"],
                "response_multiplier": scenario["response_multiplier"],
                "cost_multiplier": scenario["cost_multiplier"],
                **metrics,
            }

            row["recommended_decision"] = classify_strategy_decision(row)
            rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["scenario_name", "net_save_value_proxy", "roi_proxy"],
        ascending=[True, False, False],
    )


def classify_strategy_decision(row):
    net_value = row.get("net_save_value_proxy", 0)
    roi = row.get("roi_proxy", np.nan)
    budget_status = row.get("budget_status", "")

    roi_value = -999 if roi is None or pd.isna(roi) else roi

    if net_value > 0 and roi_value >= 10 and budget_status == "Within budget and positive ROI":
        return "Recommended campaign candidate"

    if net_value > 0 and roi_value >= 3:
        return "Good economics, review budget"

    if net_value > 0:
        return "Positive but needs testing"

    return "Do not launch without better assumptions"


def make_roi_portfolio_summary(matrix):
    base = matrix[matrix["scenario_name"].eq("Base")].copy()

    best_base = base.sort_values(
        ["net_save_value_proxy", "roi_proxy"],
        ascending=False,
    ).iloc[0]

    best_by_scenario = (
        matrix[matrix["net_save_value_proxy"] > 0]
        .sort_values(["scenario_name", "net_save_value_proxy"], ascending=[True, False])
        .groupby("scenario_name", as_index=False)
        .head(1)
    )

    row = {
        "scenario_count": matrix["scenario_name"].nunique(),
        "strategy_count": matrix["strategy_name"].nunique(),
        "best_base_strategy": best_base["strategy_name"],
        "best_base_targeted_customers": int(best_base["targeted_customers"]),
        "best_base_expected_customers_saved": float(best_base["expected_customers_saved"]),
        "best_base_expected_saved_clv_proxy": float(best_base["expected_saved_clv_proxy"]),
        "best_base_campaign_cost_proxy": float(best_base["campaign_cost_proxy"]),
        "best_base_net_save_value_proxy": float(best_base["net_save_value_proxy"]),
        "best_base_roi_proxy": float(best_base["roi_proxy"]) if not pd.isna(best_base["roi_proxy"]) else np.nan,
        "best_base_budget_status": best_base["budget_status"],
        "scenarios_with_positive_recommendation": int(
            best_by_scenario["recommended_decision"].str.contains("Recommended|Good|Positive", regex=True).sum()
        ),
    }

    return pd.DataFrame([row])


def make_best_strategy_by_scenario(matrix):
    eligible = matrix[matrix["net_save_value_proxy"] > 0].copy()

    if eligible.empty:
        return pd.DataFrame()

    eligible["budget_feasible"] = eligible["campaign_cost_proxy"] <= eligible["budget_cap"]

    feasible = eligible[eligible["budget_feasible"]].copy()

    if feasible.empty:
        feasible = eligible.copy()

    out = (
        feasible.sort_values(
            ["scenario_name", "net_save_value_proxy", "roi_proxy"],
            ascending=[True, False, False],
        )
        .groupby("scenario_name", as_index=False)
        .head(1)
        .sort_values("scenario_name")
    )

    return out


def make_action_roi_summary(df):
    rows = []

    for action, group in df.groupby("recommended_retention_action"):
        metrics = calculate_roi_metrics(
            targeted=group[group["should_receive_paid_offer"]],
            scenario_name="Base",
            response_multiplier=1.0,
            cost_multiplier=1.0,
            budget_cap=1000000,
        )

        row = {
            "recommended_retention_action": action,
            "total_customers_in_action": int(len(group)),
            "paid_offer_customers_in_action": int(group["should_receive_paid_offer"].sum()),
            **metrics,
        }

        rows.append(row)

    return pd.DataFrame(rows).sort_values("net_save_value_proxy", ascending=False)


def make_budget_cap_frontier(df):
    eligible = df[df["should_receive_paid_offer"]].copy()
    eligible = eligible.sort_values("net_save_value_proxy", ascending=False)
    eligible["base_campaign_cost"] = eligible["intervention_cost_proxy"]
    eligible["base_expected_saved_clv"] = eligible["expected_saved_clv_proxy"]

    eligible["cumulative_campaign_cost"] = eligible["base_campaign_cost"].cumsum()
    eligible["cumulative_expected_saved_clv"] = eligible["base_expected_saved_clv"].cumsum()
    eligible["cumulative_net_save_value"] = (
        eligible["cumulative_expected_saved_clv"] - eligible["cumulative_campaign_cost"]
    )

    rows = []

    for cap in BUDGET_CAPS:
        selected = eligible[eligible["cumulative_campaign_cost"] <= cap].copy()

        if selected.empty:
            rows.append(
                {
                    "budget_cap": cap,
                    "targeted_customers": 0,
                    "campaign_cost_proxy": 0,
                    "expected_saved_clv_proxy": 0,
                    "net_save_value_proxy": 0,
                    "roi_proxy": np.nan,
                    "budget_utilization": 0,
                    "marginal_readout": "Budget too small for selected offer costs.",
                }
            )
            continue

        campaign_cost = selected["base_campaign_cost"].sum()
        expected_saved = selected["base_expected_saved_clv"].sum()
        net_value = expected_saved - campaign_cost
        roi = net_value / campaign_cost if campaign_cost > 0 else np.nan

        rows.append(
            {
                "budget_cap": cap,
                "targeted_customers": int(len(selected)),
                "campaign_cost_proxy": float(campaign_cost),
                "expected_saved_clv_proxy": float(expected_saved),
                "net_save_value_proxy": float(net_value),
                "roi_proxy": None if pd.isna(roi) else float(roi),
                "budget_utilization": float(campaign_cost / cap) if cap > 0 else np.nan,
                "marginal_readout": "Budget frontier point ranked by save-worthiness.",
            }
        )

    out = pd.DataFrame(rows)
    out["incremental_customers"] = out["targeted_customers"].diff().fillna(out["targeted_customers"])
    out["incremental_net_value"] = out["net_save_value_proxy"].diff().fillna(out["net_save_value_proxy"])

    return out


def make_response_rate_sensitivity(df):
    eligible = df[df["should_receive_paid_offer"]].copy()

    strategy_filters = {
        "Tier 1 plus Tier 2 premium save": eligible["save_priority_tier"].isin(
            [
                "Tier 1 - Executive save priority",
                "Tier 2 - Premium save priority",
            ]
        ),
        "Tiers 1 to 3 targeted save": eligible["save_priority_tier"].isin(
            [
                "Tier 1 - Executive save priority",
                "Tier 2 - Premium save priority",
                "Tier 3 - Targeted save priority",
            ]
        ),
        "All positive ROI paid-offer candidates": eligible["should_receive_paid_offer"],
    }

    rows = []

    for strategy_name, mask in strategy_filters.items():
        targeted = eligible[mask].copy()

        for multiplier in RESPONSE_MULTIPLIERS:
            metrics = calculate_roi_metrics(
                targeted=targeted,
                scenario_name=f"Response multiplier {multiplier:.2f}",
                response_multiplier=multiplier,
                cost_multiplier=1.0,
                budget_cap=1000000,
            )

            row = {
                "strategy_name": strategy_name,
                "response_multiplier": multiplier,
                **metrics,
            }

            rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["strategy_name", "response_multiplier"]
    )


def make_break_even_summary(matrix):
    out = matrix.copy()

    out["break_even_response_rate_display"] = out["break_even_response_rate"]
    out["response_headroom"] = (
        out["adjusted_expected_response_rate"] - out["break_even_response_rate"]
    )

    out["break_even_readout"] = np.select(
        [
            out["break_even_response_rate"].isna(),
            out["response_headroom"] >= 0.05,
            out["response_headroom"] >= 0.01,
            out["response_headroom"] >= 0,
            out["response_headroom"] < 0,
        ],
        [
            "No paid campaign cost",
            "Comfortable response-rate cushion",
            "Positive but narrow response-rate cushion",
            "Near break-even",
            "Below break-even",
        ],
        default="Review",
    )

    return out[
        [
            "strategy_name",
            "scenario_name",
            "targeted_customers",
            "adjusted_expected_response_rate",
            "break_even_response_rate",
            "response_headroom",
            "campaign_cost_proxy",
            "expected_saved_clv_proxy",
            "net_save_value_proxy",
            "roi_proxy",
            "break_even_readout",
        ]
    ].sort_values(["scenario_name", "net_save_value_proxy"], ascending=[True, False])


def make_tableau_roi_frontier(matrix):
    out = matrix.copy()

    keep_cols = [
        "strategy_name",
        "strategy_type",
        "scenario_name",
        "targeted_customers",
        "expected_customers_saved",
        "expected_saved_clv_proxy",
        "campaign_cost_proxy",
        "net_save_value_proxy",
        "roi_proxy",
        "break_even_response_rate",
        "budget_cap",
        "budget_utilization",
        "budget_status",
        "recommended_decision",
    ]

    return out[keep_cols].copy()


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

    matrix = outputs["01_strategy_scenario_matrix"]
    summary = outputs["00_roi_portfolio_summary"].iloc[0]
    best = outputs["05_best_strategy_by_scenario"]
    frontier = outputs["03_budget_cap_frontier"]

    add_check(
        "strategy_scenario_matrix_not_empty",
        len(matrix),
        len(matrix) > 0,
        "ROI matrix should contain strategy/scenario rows.",
    )

    add_check(
        "base_best_strategy_positive_net_value",
        round(float(summary["best_base_net_save_value_proxy"]), 2),
        float(summary["best_base_net_save_value_proxy"]) > 0,
        "Best base strategy should have positive net value.",
    )

    add_check(
        "base_best_strategy_positive_roi",
        round(float(summary["best_base_roi_proxy"]), 6),
        float(summary["best_base_roi_proxy"]) > 0,
        "Best base strategy should have positive ROI.",
    )

    add_check(
        "best_strategy_by_scenario_created",
        len(best),
        len(best) > 0,
        "There should be at least one best strategy row by scenario.",
    )

    add_check(
        "budget_frontier_created",
        len(frontier),
        len(frontier) == len(BUDGET_CAPS),
        "Budget frontier should have one row per budget cap.",
    )

    add_check(
        "no_negative_campaign_cost",
        round(float(matrix["campaign_cost_proxy"].min()), 6),
        matrix["campaign_cost_proxy"].min() >= 0,
        "Campaign costs should not be negative.",
    )

    report = pd.DataFrame(checks)
    report.to_csv(OUTPUT_DIR / "_retention_roi_validation_report.csv", index=False)

    with open(OUTPUT_DIR / "_retention_roi_validation_report.json", "w") as f:
        json.dump(clean_for_json(checks), f, indent=2)

    print("\nValidation report:")
    print(report.to_string(index=False))

    failed = report[report["status"] == "FAIL"]
    if len(failed) > 0:
        raise ValueError("Retention ROI validation failed. Review outputs before moving forward.")

    return report


def write_sql_reference():
    sql_path = SQL_DIR / "09_retention_roi_simulation.sql"

    lines = [
        "-- 09_retention_roi_simulation.sql",
        "-- SQL reference for retention ROI simulation outputs.",
        "",
        "-- Strategy/scenario matrix",
        "SELECT",
        "    strategy_name,",
        "    scenario_name,",
        "    targeted_customers,",
        "    expected_customers_saved,",
        "    expected_saved_clv_proxy,",
        "    campaign_cost_proxy,",
        "    net_save_value_proxy,",
        "    roi_proxy,",
        "    budget_status,",
        "    recommended_decision",
        "FROM read_csv_auto('data/processed/retention_roi_outputs/01_strategy_scenario_matrix.csv')",
        "ORDER BY scenario_name, net_save_value_proxy DESC;",
        "",
        "-- Budget frontier",
        "SELECT",
        "    budget_cap,",
        "    targeted_customers,",
        "    campaign_cost_proxy,",
        "    expected_saved_clv_proxy,",
        "    net_save_value_proxy,",
        "    roi_proxy",
        "FROM read_csv_auto('data/processed/retention_roi_outputs/03_budget_cap_frontier.csv')",
        "ORDER BY budget_cap;",
        "",
        "-- Break-even response analysis",
        "SELECT",
        "    strategy_name,",
        "    scenario_name,",
        "    adjusted_expected_response_rate,",
        "    break_even_response_rate,",
        "    response_headroom,",
        "    break_even_readout",
        "FROM read_csv_auto('data/processed/retention_roi_outputs/07_break_even_summary.csv')",
        "ORDER BY scenario_name, net_save_value_proxy DESC;",
    ]

    with open(sql_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved SQL reference file: {sql_path}")


def write_executive_summary(outputs, validation_report):
    summary = outputs["00_roi_portfolio_summary"].iloc[0]
    best_by_scenario = outputs["05_best_strategy_by_scenario"]
    frontier = outputs["03_budget_cap_frontier"]
    break_even = outputs["07_break_even_summary"]

    best_base = best_by_scenario[best_by_scenario["scenario_name"].eq("Base")]

    if best_base.empty:
        best_base = outputs["01_strategy_scenario_matrix"][
            outputs["01_strategy_scenario_matrix"]["scenario_name"].eq("Base")
        ].sort_values("net_save_value_proxy", ascending=False).head(1)

    best_base = best_base.iloc[0]

    best_budget = frontier.sort_values("net_save_value_proxy", ascending=False).iloc[0]
    best_break_even = break_even.sort_values("net_save_value_proxy", ascending=False).iloc[0]

    efficient_frontier = frontier.copy()
    max_net_value = efficient_frontier["net_save_value_proxy"].max()
    efficient_frontier["net_value_capture_vs_max"] = (
        efficient_frontier["net_save_value_proxy"] / max_net_value
        if max_net_value > 0
        else 0
    )

    efficient_budget = (
        efficient_frontier[efficient_frontier["net_value_capture_vs_max"] >= 0.99]
        .sort_values("budget_cap")
        .iloc[0]
    )

    executive = {
        "best_base_strategy": str(best_base["strategy_name"]),
        "best_base_targeted_customers": int(best_base["targeted_customers"]),
        "best_base_expected_customers_saved": float(best_base["expected_customers_saved"]),
        "best_base_expected_saved_clv_proxy": float(best_base["expected_saved_clv_proxy"]),
        "best_base_campaign_cost_proxy": float(best_base["campaign_cost_proxy"]),
        "best_base_net_save_value_proxy": float(best_base["net_save_value_proxy"]),
        "best_base_roi_proxy": float(best_base["roi_proxy"]) if not pd.isna(best_base["roi_proxy"]) else None,
        "best_base_budget_status": str(best_base["budget_status"]),
        "best_budget_cap": float(best_budget["budget_cap"]),
        "best_budget_targeted_customers": int(best_budget["targeted_customers"]),
        "best_budget_net_save_value_proxy": float(best_budget["net_save_value_proxy"]),
        "best_budget_roi_proxy": float(best_budget["roi_proxy"]) if not pd.isna(best_budget["roi_proxy"]) else None,
        "efficient_budget_cap": float(efficient_budget["budget_cap"]),
        "efficient_budget_targeted_customers": int(efficient_budget["targeted_customers"]),
        "efficient_budget_net_save_value_proxy": float(efficient_budget["net_save_value_proxy"]),
        "efficient_budget_value_capture_vs_max": float(efficient_budget["net_value_capture_vs_max"]),
        "break_even_response_rate": (
            None
            if pd.isna(best_break_even["break_even_response_rate"])
            else float(best_break_even["break_even_response_rate"])
        ),
        "validation_status": "PASS" if (validation_report["status"] == "PASS").all() else "FAIL",
    }

    with open(OUTPUT_DIR / "_retention_roi_summary.json", "w") as f:
        json.dump(clean_for_json(executive), f, indent=2)

    roi_line = (
        f"{executive['best_base_roi_proxy']:.2f}x"
        if executive["best_base_roi_proxy"] is not None
        else "N/A"
    )

    break_even_line = (
        f"{executive['break_even_response_rate']:.2%}"
        if executive["break_even_response_rate"] is not None
        else "N/A"
    )

    lines = [
        "# Retention ROI Simulation Summary",
        "",
        f"Best base-case strategy: {executive['best_base_strategy']}",
        f"Targeted customers: {executive['best_base_targeted_customers']:,}",
        f"Expected customers saved: {executive['best_base_expected_customers_saved']:,.0f}",
        f"Expected saved CLV proxy: {executive['best_base_expected_saved_clv_proxy']:,.0f}",
        f"Campaign cost proxy: {executive['best_base_campaign_cost_proxy']:,.0f}",
        f"Expected net save value proxy: {executive['best_base_net_save_value_proxy']:,.0f}",
        f"ROI proxy: {roi_line}",
        f"Budget status: {executive['best_base_budget_status']}",
        "",
        "Budget frontier readout:",
        (
            f"- The budget cap that maximizes absolute expected net value is {executive['best_budget_cap']:,.0f}, "
            f"targeting {executive['best_budget_targeted_customers']:,} customers and producing "
            f"{executive['best_budget_net_save_value_proxy']:,.0f} expected net save value proxy."
        ),
        (
            f"- The efficient frontier point is approximately {executive['efficient_budget_cap']:,.0f}, "
            f"which captures {executive['efficient_budget_value_capture_vs_max']:.2%} of the maximum tested net value "
            f"while targeting {executive['efficient_budget_targeted_customers']:,} customers."
        ),
        (
            "- This shows diminishing returns after the efficient frontier point, so leadership can separate "
            "maximum-value planning from budget-efficient campaign sizing."
        ),
        "",
        "Break-even readout:",
        f"- Best-strategy break-even response rate: {break_even_line}",
        "",
        "Business interpretation:",
        (
            "The retention engine should not launch a blanket campaign. The best economics come from "
            "targeting a focused save-worthy population and stress-testing the response assumptions before rollout. "
            "This script gives leadership a campaign frontier: target size, budget, expected saved value, cost, ROI, "
            "and break-even response rate."
        ),
        "",
        "Next step:",
        (
            "The next script should design the A/B test: treatment/control sizing, measurable lift, expected power, "
            "success metrics, and decision rules for rollout."
        ),
        "",
        "Assumption note:",
        (
            "This ROI simulation uses planning assumptions from the save-worthiness layer. Actual response rates "
            "must be validated through an experiment before full deployment."
        ),
    ]

    summary_path = OUTPUT_DIR / "_executive_retention_roi_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved {summary_path}")
    print(f"Saved {OUTPUT_DIR / '_retention_roi_summary.json'}")


def main():
    print("\nRunning retention ROI simulation...")

    df = load_save_worthiness_scores()

    matrix = simulate_strategy_scenario_matrix(df)
    outputs = {
        "01_strategy_scenario_matrix": matrix,
        "00_roi_portfolio_summary": make_roi_portfolio_summary(matrix),
        "02_action_roi_summary": make_action_roi_summary(df),
        "03_budget_cap_frontier": make_budget_cap_frontier(df),
        "04_response_rate_sensitivity": make_response_rate_sensitivity(df),
        "05_best_strategy_by_scenario": make_best_strategy_by_scenario(matrix),
        "06_tableau_retention_roi_frontier": make_tableau_roi_frontier(matrix),
        "07_break_even_summary": make_break_even_summary(matrix),
    }

    write_sql_reference()

    with open(OUTPUT_DIR / "_retention_roi_assumptions.json", "w") as f:
        json.dump(
            {
                "roi_scenarios": ROI_SCENARIOS,
                "budget_caps": BUDGET_CAPS,
                "response_multipliers": RESPONSE_MULTIPLIERS,
                "note": "Planning assumptions for ROI simulation. Final response rates require A/B testing.",
            },
            f,
            indent=2,
        )

    for name, out in outputs.items():
        path = OUTPUT_DIR / f"{name}.csv"
        out.to_csv(path, index=False)
        print(f"Saved {path} | rows={len(out):,} cols={out.shape[1]:,}")

    validation_report = validate_outputs(outputs)
    write_executive_summary(outputs, validation_report)

    print("\n09_retention_roi_simulation.py complete.")


if __name__ == "__main__":
    main()
