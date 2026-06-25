"""
04_revenue_leakage_analysis.py

Revenue leakage and value concentration layer for the Customer Revenue Recovery
& Retention ROI Engine.

This is the bridge between descriptive retention analysis and predictive modeling.

By this point, the project has:
1. A customer-month model
2. Cohort retention outputs
3. A profit-adjusted CLV layer

This script answers the business question I would want answered before building
a churn model:

Where is value actually leaking, how concentrated is the exposure, and which
customer groups deserve model-driven retention decisions?

Important note:
The KKBox churn label is a future outcome. I use it here for retrospective leakage
diagnosis and opportunity sizing, not for final targeting. Final targeting happens
later after the churn model creates predicted churn probabilities.
"""

from pathlib import Path
import json
import duckdb
import pandas as pd


PROCESSED_DIR = Path("data/processed")
CLV_DIR = PROCESSED_DIR / "clv_outputs"
OUTPUT_DIR = PROCESSED_DIR / "revenue_leakage_outputs"
SQL_DIR = Path("sql")

CLV_PATH = CLV_DIR / "customer_clv_scores.parquet"
CUSTOMER_MONTH_PATH = PROCESSED_DIR / "customer_month_table.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SQL_DIR.mkdir(parents=True, exist_ok=True)


ASSUMPTIONS = {
    "recovery_scenarios": {
        "Conservative recovery": 0.05,
        "Base recovery": 0.10,
        "Aggressive recovery": 0.15,
    },
    "campaign_cost_per_customer": {
        "Premium save budget": 15.00,
        "Moderate save budget": 6.00,
        "Low-cost save budget": 2.00,
        "Automation only": 0.50,
        "No paid budget": 0.00,
    },
    "expected_recovery_rate_by_budget": {
        "Premium save budget": 0.12,
        "Moderate save budget": 0.06,
        "Low-cost save budget": 0.03,
        "Automation only": 0.01,
        "No paid budget": 0.00,
    },
    "note": (
        "These are planning assumptions for opportunity sizing. They are not final "
        "campaign results. The ROI simulator later will allow these assumptions to vary."
    ),
}


SQL_QUERIES = {
    "00_revenue_leakage_portfolio_summary": """
        SELECT
            COUNT(*) AS customers,
            AVG(churn_next_period) AS future_churn_rate,
            SUM(monthly_value_baseline) AS monthly_revenue_baseline_proxy,
            SUM(monthly_margin_baseline) AS monthly_margin_baseline_proxy,
            SUM(annual_revenue_run_rate_proxy) AS annual_revenue_run_rate_proxy,
            SUM(annual_margin_run_rate_proxy) AS annual_margin_run_rate_proxy,
            SUM(profit_adjusted_clv_proxy) AS profit_adjusted_clv_proxy,
            SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
            CASE
                WHEN SUM(profit_adjusted_clv_proxy) > 0
                THEN SUM(future_churned_clv_proxy) / SUM(profit_adjusted_clv_proxy)
                ELSE NULL
            END AS clv_leakage_rate,
            SUM(CASE WHEN churn_next_period = 1 THEN monthly_value_baseline ELSE 0 END)
                AS future_churned_monthly_revenue_proxy,
            SUM(CASE WHEN churn_next_period = 1 THEN monthly_margin_baseline ELSE 0 END)
                AS future_churned_monthly_margin_proxy
        FROM customer_clv
    """,

    "01_value_leakage_by_clv_tier": """
        SELECT
            clv_value_tier,
            retention_budget_tier,
            COUNT(*) AS customers,
            AVG(churn_next_period) AS future_churn_rate,
            SUM(monthly_value_baseline) AS monthly_revenue_baseline_proxy,
            SUM(monthly_margin_baseline) AS monthly_margin_baseline_proxy,
            SUM(profit_adjusted_clv_proxy) AS profit_adjusted_clv_proxy,
            SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
            CASE
                WHEN SUM(profit_adjusted_clv_proxy) > 0
                THEN SUM(future_churned_clv_proxy) / SUM(profit_adjusted_clv_proxy)
                ELSE NULL
            END AS clv_leakage_rate,
            AVG(engagement_score) AS avg_engagement_score,
            AVG(auto_renew_signal) AS auto_renew_rate,
            AVG(cancellation_signal) AS cancellation_signal_rate
        FROM customer_clv
        GROUP BY
            clv_value_tier,
            retention_budget_tier
        ORDER BY future_churned_clv_proxy DESC
    """,

    "02_value_leakage_by_action_group": """
        SELECT
            value_based_action_group,
            retention_budget_tier,
            COUNT(*) AS customers,
            AVG(churn_next_period) AS future_churn_rate,
            SUM(monthly_value_baseline) AS monthly_revenue_baseline_proxy,
            SUM(monthly_margin_baseline) AS monthly_margin_baseline_proxy,
            SUM(profit_adjusted_clv_proxy) AS profit_adjusted_clv_proxy,
            SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
            CASE
                WHEN SUM(profit_adjusted_clv_proxy) > 0
                THEN SUM(future_churned_clv_proxy) / SUM(profit_adjusted_clv_proxy)
                ELSE NULL
            END AS clv_leakage_rate,
            AVG(engagement_score) AS avg_engagement_score,
            AVG(auto_renew_signal) AS auto_renew_rate,
            AVG(cancellation_signal) AS cancellation_signal_rate,
            AVG(no_recent_activity_flag) AS no_recent_activity_rate,
            AVG(major_activity_drop_flag) AS major_activity_drop_rate
        FROM customer_clv
        GROUP BY
            value_based_action_group,
            retention_budget_tier
        ORDER BY future_churned_clv_proxy DESC
    """,

    "03_value_leakage_segment_matrix": """
        SELECT
            lifecycle_stage,
            engagement_tier,
            revenue_tier,
            clv_value_tier,
            value_based_action_group,
            retention_budget_tier,
            COUNT(*) AS customers,
            AVG(churn_next_period) AS future_churn_rate,
            SUM(monthly_value_baseline) AS monthly_revenue_baseline_proxy,
            SUM(profit_adjusted_clv_proxy) AS profit_adjusted_clv_proxy,
            SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
            CASE
                WHEN SUM(profit_adjusted_clv_proxy) > 0
                THEN SUM(future_churned_clv_proxy) / SUM(profit_adjusted_clv_proxy)
                ELSE NULL
            END AS clv_leakage_rate,
            AVG(engagement_score) AS avg_engagement_score,
            AVG(active_days) AS avg_active_days,
            AVG(auto_renew_signal) AS auto_renew_rate,
            AVG(cancellation_signal) AS cancellation_signal_rate,
            AVG(no_recent_activity_flag) AS no_recent_activity_rate,
            AVG(major_activity_drop_flag) AS major_activity_drop_rate
        FROM customer_clv
        GROUP BY
            lifecycle_stage,
            engagement_tier,
            revenue_tier,
            clv_value_tier,
            value_based_action_group,
            retention_budget_tier
        HAVING COUNT(*) >= 500
        ORDER BY future_churned_clv_proxy DESC
    """,

    "04_clv_concentration_deciles": """
        WITH ranked AS (
            SELECT
                *,
                NTILE(10) OVER (ORDER BY profit_adjusted_clv_proxy DESC) AS clv_rank_decile
            FROM customer_clv
        ),

        decile_summary AS (
            SELECT
                clv_rank_decile,
                COUNT(*) AS customers,
                AVG(churn_next_period) AS future_churn_rate,
                SUM(profit_adjusted_clv_proxy) AS profit_adjusted_clv_proxy,
                SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
                AVG(profit_adjusted_clv_proxy) AS avg_profit_adjusted_clv_proxy,
                AVG(monthly_value_baseline) AS avg_monthly_value_baseline,
                AVG(engagement_score) AS avg_engagement_score,
                AVG(auto_renew_signal) AS auto_renew_rate,
                AVG(cancellation_signal) AS cancellation_signal_rate
            FROM ranked
            GROUP BY clv_rank_decile
        ),

        with_shares AS (
            SELECT
                *,
                profit_adjusted_clv_proxy
                    / NULLIF(SUM(profit_adjusted_clv_proxy) OVER (), 0)
                    AS clv_share,
                future_churned_clv_proxy
                    / NULLIF(SUM(future_churned_clv_proxy) OVER (), 0)
                    AS future_churned_clv_share
            FROM decile_summary
        )

        SELECT
            *,
            SUM(clv_share) OVER (
                ORDER BY clv_rank_decile
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cumulative_clv_share,
            SUM(future_churned_clv_share) OVER (
                ORDER BY clv_rank_decile
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cumulative_future_churned_clv_share
        FROM with_shares
        ORDER BY clv_rank_decile
    """,

    "05_revenue_leakage_driver_scores": """
        WITH segment_base AS (
            SELECT
                lifecycle_stage,
                engagement_tier,
                revenue_tier,
                clv_value_tier,
                value_based_action_group,
                retention_budget_tier,
                COUNT(*) AS customers,
                AVG(churn_next_period) AS future_churn_rate,
                SUM(profit_adjusted_clv_proxy) AS profit_adjusted_clv_proxy,
                SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
                AVG(engagement_score) AS avg_engagement_score,
                AVG(active_days) AS avg_active_days,
                AVG(auto_renew_signal) AS auto_renew_rate,
                AVG(cancellation_signal) AS cancellation_signal_rate,
                AVG(no_recent_activity_flag) AS no_recent_activity_rate,
                AVG(major_activity_drop_flag) AS major_activity_drop_rate
            FROM customer_clv
            GROUP BY
                lifecycle_stage,
                engagement_tier,
                revenue_tier,
                clv_value_tier,
                value_based_action_group,
                retention_budget_tier
            HAVING COUNT(*) >= 500
        )

        SELECT
            *,
            future_churned_clv_proxy
                * (1 + cancellation_signal_rate)
                * (1 + major_activity_drop_rate)
                AS leakage_priority_score,

            CASE
                WHEN retention_budget_tier = 'Premium save budget'
                    AND future_churn_rate >= 0.10
                    THEN 'Model should prioritize'
                WHEN value_based_action_group = 'Suppress paid offer'
                    THEN 'Suppress from paid retention'
                WHEN cancellation_signal_rate >= 0.50
                    THEN 'Win-back strategy needed'
                WHEN no_recent_activity_rate >= 0.50
                    THEN 'Reactivation strategy needed'
                WHEN future_churn_rate < 0.03
                    THEN 'Monitor, do not overspend'
                ELSE 'Watchlist segment'
            END AS leakage_strategy_readout
        FROM segment_base
        ORDER BY leakage_priority_score DESC
    """,

    "06_retention_budget_pressure": """
        SELECT
            retention_budget_tier,
            COUNT(*) AS customers,
            AVG(churn_next_period) AS future_churn_rate,
            SUM(monthly_value_baseline) AS monthly_revenue_baseline_proxy,
            SUM(monthly_margin_baseline) AS monthly_margin_baseline_proxy,
            SUM(profit_adjusted_clv_proxy) AS profit_adjusted_clv_proxy,
            SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
            CASE
                WHEN SUM(profit_adjusted_clv_proxy) > 0
                THEN SUM(future_churned_clv_proxy) / SUM(profit_adjusted_clv_proxy)
                ELSE NULL
            END AS clv_leakage_rate,
            AVG(engagement_score) AS avg_engagement_score,
            AVG(cancellation_signal) AS cancellation_signal_rate
        FROM customer_clv
        GROUP BY retention_budget_tier
        ORDER BY future_churned_clv_proxy DESC
    """,

    "07_monthly_value_leakage_trend": """
        SELECT
            cm.snapshot_month,
            COUNT(DISTINCT cm.msno) AS observed_customers,
            SUM(cm.monthly_revenue) AS observed_monthly_revenue_proxy,
            SUM(cm.monthly_revenue * c.gross_margin_rate) AS observed_monthly_margin_proxy,
            AVG(cm.churn_next_period) AS future_churn_label_rate,
            SUM(CASE WHEN cm.churn_next_period = 1 THEN cm.monthly_revenue ELSE 0 END)
                AS future_churned_monthly_revenue_proxy,
            SUM(CASE WHEN cm.churn_next_period = 1 THEN cm.monthly_revenue * c.gross_margin_rate ELSE 0 END)
                AS future_churned_monthly_margin_proxy,
            AVG(cm.engagement_score) AS avg_engagement_score,
            AVG(cm.had_cancel) AS cancel_signal_rate
        FROM customer_month cm
        LEFT JOIN customer_clv c
            ON cm.msno = c.msno
        GROUP BY cm.snapshot_month
        ORDER BY cm.snapshot_month
    """,

    "08_tableau_revenue_leakage_base": """
        SELECT
            lifecycle_stage,
            engagement_tier,
            revenue_tier,
            clv_value_tier,
            value_based_action_group,
            retention_budget_tier,
            COUNT(*) AS customers,
            AVG(churn_next_period) AS future_churn_rate,
            SUM(monthly_value_baseline) AS monthly_revenue_baseline_proxy,
            SUM(profit_adjusted_clv_proxy) AS profit_adjusted_clv_proxy,
            SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
            CASE
                WHEN SUM(profit_adjusted_clv_proxy) > 0
                THEN SUM(future_churned_clv_proxy) / SUM(profit_adjusted_clv_proxy)
                ELSE NULL
            END AS clv_leakage_rate,
            AVG(engagement_score) AS avg_engagement_score,
            AVG(active_days) AS avg_active_days,
            AVG(auto_renew_signal) AS auto_renew_rate,
            AVG(cancellation_signal) AS cancellation_signal_rate,
            AVG(no_recent_activity_flag) AS no_recent_activity_rate,
            AVG(major_activity_drop_flag) AS major_activity_drop_rate
        FROM customer_clv
        GROUP BY
            lifecycle_stage,
            engagement_tier,
            revenue_tier,
            clv_value_tier,
            value_based_action_group,
            retention_budget_tier
        HAVING COUNT(*) >= 250
        ORDER BY future_churned_clv_proxy DESC
    """,

    "09_value_concentration_thresholds": """
        WITH ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (ORDER BY profit_adjusted_clv_proxy DESC) AS value_rank,
                COUNT(*) OVER () AS total_customers,
                SUM(profit_adjusted_clv_proxy) OVER () AS portfolio_clv_proxy,
                SUM(future_churned_clv_proxy) OVER () AS portfolio_future_churned_clv_proxy
            FROM customer_clv
        ),

        thresholds AS (
            SELECT *
            FROM (
                VALUES
                    ('Top 1%', 0.01),
                    ('Top 5%', 0.05),
                    ('Top 10%', 0.10),
                    ('Top 20%', 0.20)
            ) AS t(threshold_label, threshold_pct)
        )

        SELECT
            t.threshold_label,
            t.threshold_pct,
            COUNT(*) AS customers,
            SUM(r.profit_adjusted_clv_proxy) AS profit_adjusted_clv_proxy,
            SUM(r.future_churned_clv_proxy) AS future_churned_clv_proxy,
            SUM(r.profit_adjusted_clv_proxy) / NULLIF(MAX(r.portfolio_clv_proxy), 0)
                AS clv_share,
            SUM(r.future_churned_clv_proxy) / NULLIF(MAX(r.portfolio_future_churned_clv_proxy), 0)
                AS future_churned_clv_share,
            AVG(r.churn_next_period) AS future_churn_rate,
            AVG(r.monthly_value_baseline) AS avg_monthly_value_baseline,
            AVG(r.engagement_score) AS avg_engagement_score
        FROM ranked r
        INNER JOIN thresholds t
            ON r.value_rank <= CEIL(r.total_customers * t.threshold_pct)
        GROUP BY
            t.threshold_label,
            t.threshold_pct
        ORDER BY t.threshold_pct
    """,

    "10_recoverable_value_scenarios": """
        WITH portfolio AS (
            SELECT
                SUM(future_churned_clv_proxy) AS portfolio_future_churned_clv_proxy,
                SUM(CASE WHEN retention_budget_tier = 'Premium save budget'
                    THEN future_churned_clv_proxy ELSE 0 END) AS premium_budget_future_churned_clv_proxy,
                SUM(CASE WHEN retention_budget_tier IN ('Premium save budget', 'Moderate save budget')
                    THEN future_churned_clv_proxy ELSE 0 END) AS premium_plus_moderate_future_churned_clv_proxy
            FROM customer_clv
        ),

        scenarios AS (
            SELECT *
            FROM (
                VALUES
                    ('Conservative recovery', 0.05),
                    ('Base recovery', 0.10),
                    ('Aggressive recovery', 0.15)
            ) AS s(scenario_name, recovery_rate)
        )

        SELECT
            s.scenario_name,
            s.recovery_rate,
            p.portfolio_future_churned_clv_proxy,
            p.portfolio_future_churned_clv_proxy * s.recovery_rate
                AS recoverable_portfolio_clv_proxy,
            p.premium_budget_future_churned_clv_proxy,
            p.premium_budget_future_churned_clv_proxy * s.recovery_rate
                AS recoverable_premium_budget_clv_proxy,
            p.premium_plus_moderate_future_churned_clv_proxy,
            p.premium_plus_moderate_future_churned_clv_proxy * s.recovery_rate
                AS recoverable_premium_plus_moderate_clv_proxy
        FROM portfolio p
        CROSS JOIN scenarios s
        ORDER BY s.recovery_rate
    """,

    "11_retention_budget_opportunity": """
        WITH budget_base AS (
            SELECT
                retention_budget_tier,
                COUNT(*) AS customers,
                AVG(churn_next_period) AS future_churn_rate,
                SUM(profit_adjusted_clv_proxy) AS profit_adjusted_clv_proxy,
                SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
                AVG(engagement_score) AS avg_engagement_score,
                AVG(cancellation_signal) AS cancellation_signal_rate
            FROM customer_clv
            GROUP BY retention_budget_tier
        ),

        assumptions AS (
            SELECT
                *,
                CASE retention_budget_tier
                    WHEN 'Premium save budget' THEN 15.00
                    WHEN 'Moderate save budget' THEN 6.00
                    WHEN 'Low-cost save budget' THEN 2.00
                    WHEN 'Automation only' THEN 0.50
                    ELSE 0.00
                END AS campaign_cost_per_customer,

                CASE retention_budget_tier
                    WHEN 'Premium save budget' THEN 0.12
                    WHEN 'Moderate save budget' THEN 0.06
                    WHEN 'Low-cost save budget' THEN 0.03
                    WHEN 'Automation only' THEN 0.01
                    ELSE 0.00
                END AS expected_recovery_rate
            FROM budget_base
        ),

        scored AS (
            SELECT
                *,
                customers * campaign_cost_per_customer AS campaign_cost_proxy,
                future_churned_clv_proxy * expected_recovery_rate AS recoverable_clv_proxy,
                future_churned_clv_proxy * expected_recovery_rate
                    - customers * campaign_cost_per_customer AS net_recovery_opportunity_proxy,
                CASE
                    WHEN future_churned_clv_proxy > 0
                    THEN (customers * campaign_cost_per_customer) / future_churned_clv_proxy
                    ELSE NULL
                END AS break_even_recovery_rate
            FROM assumptions
        )

        SELECT
            *,
            CASE
                WHEN campaign_cost_proxy > 0
                THEN net_recovery_opportunity_proxy / campaign_cost_proxy
                ELSE NULL
            END AS roi_proxy
        FROM scored
        ORDER BY net_recovery_opportunity_proxy DESC
    """,

    "12_model_priority_segments": """
        WITH segment_base AS (
            SELECT
                lifecycle_stage,
                engagement_tier,
                revenue_tier,
                clv_value_tier,
                value_based_action_group,
                retention_budget_tier,
                COUNT(*) AS customers,
                AVG(churn_next_period) AS future_churn_rate,
                SUM(profit_adjusted_clv_proxy) AS profit_adjusted_clv_proxy,
                SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
                AVG(monthly_value_baseline) AS avg_monthly_value_baseline,
                AVG(engagement_score) AS avg_engagement_score,
                AVG(auto_renew_signal) AS auto_renew_rate,
                AVG(cancellation_signal) AS cancellation_signal_rate,
                AVG(no_recent_activity_flag) AS no_recent_activity_rate
            FROM customer_clv
            GROUP BY
                lifecycle_stage,
                engagement_tier,
                revenue_tier,
                clv_value_tier,
                value_based_action_group,
                retention_budget_tier
            HAVING COUNT(*) >= 500
        )

        SELECT
            *,
            future_churned_clv_proxy
                * (1 + future_churn_rate)
                * CASE
                    WHEN retention_budget_tier = 'Premium save budget' THEN 1.50
                    WHEN retention_budget_tier = 'Moderate save budget' THEN 1.20
                    WHEN retention_budget_tier = 'Low-cost save budget' THEN 0.80
                    WHEN retention_budget_tier = 'Automation only' THEN 0.50
                    ELSE 0.00
                  END AS model_priority_score,

            CASE
                WHEN retention_budget_tier = 'Premium save budget'
                    AND future_churned_clv_proxy > 1000000
                    THEN 'Highest priority for churn model'
                WHEN retention_budget_tier = 'Moderate save budget'
                    AND future_churned_clv_proxy > 1000000
                    THEN 'Secondary model priority'
                WHEN value_based_action_group = 'Suppress paid offer'
                    THEN 'Exclude from paid-retention model action'
                WHEN retention_budget_tier = 'Automation only'
                    THEN 'Use for low-cost lifecycle automation'
                ELSE 'Monitor segment'
            END AS model_priority_recommendation,

            CASE
                WHEN retention_budget_tier = 'Premium save budget'
                    THEN 'High value exposure; model should separate customers who need intervention from customers who would stay anyway.'
                WHEN retention_budget_tier = 'Moderate save budget'
                    THEN 'Moderate value exposure; model can improve campaign efficiency.'
                WHEN retention_budget_tier = 'Automation only'
                    THEN 'Low individual value; use scalable low-cost messaging.'
                WHEN value_based_action_group = 'Suppress paid offer'
                    THEN 'No observed value; paid offers should be avoided.'
                ELSE 'Use as monitoring population.'
            END AS modeling_reason
        FROM segment_base
        ORDER BY model_priority_score DESC
    """,
}


def write_sql_file() -> None:
    sql_path = SQL_DIR / "04_revenue_leakage_analysis.sql"

    with open(sql_path, "w") as f:
        for name, query in SQL_QUERIES.items():
            f.write(f"-- {name}\n")
            f.write(query.strip())
            f.write(";\n\n")

    print(f"Saved SQL reference file: {sql_path}")


def export_query(con: duckdb.DuckDBPyConnection, name: str, query: str) -> pd.DataFrame:
    df = con.execute(query).df()
    output_path = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} | rows={len(df):,} cols={df.shape[1]:,}")
    return df


def validate_outputs(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    checks = []

    def add_check(check_name: str, value, passed: bool, notes: str) -> None:
        checks.append(
            {
                "check_name": check_name,
                "status": "PASS if passed else FAIL",
                "value": value,
                "notes": notes,
            }
        )
        checks[-1]["status"] = "PASS" if passed else "FAIL"

    portfolio = results["00_revenue_leakage_portfolio_summary"].iloc[0]
    concentration = results["04_clv_concentration_deciles"]
    thresholds = results["09_value_concentration_thresholds"]
    scenarios = results["10_recoverable_value_scenarios"]
    budget_opportunity = results["11_retention_budget_opportunity"]
    model_priority = results["12_model_priority_segments"]

    add_check(
        "portfolio_summary_has_one_row",
        len(results["00_revenue_leakage_portfolio_summary"]),
        len(results["00_revenue_leakage_portfolio_summary"]) == 1,
        "Portfolio summary should return one row.",
    )

    add_check(
        "positive_profit_adjusted_clv",
        round(float(portfolio["profit_adjusted_clv_proxy"]), 2),
        float(portfolio["profit_adjusted_clv_proxy"]) > 0,
        "Total CLV proxy should be positive.",
    )

    add_check(
        "positive_future_churned_clv",
        round(float(portfolio["future_churned_clv_proxy"]), 2),
        float(portfolio["future_churned_clv_proxy"]) > 0,
        "Future churned CLV proxy should be positive for leakage analysis.",
    )

    add_check(
        "leakage_rate_reasonable",
        round(float(portfolio["clv_leakage_rate"]), 6),
        0 <= float(portfolio["clv_leakage_rate"]) <= 1,
        "Leakage rate should be between 0 and 100%.",
    )

    add_check(
        "clv_concentration_has_10_deciles",
        concentration["clv_rank_decile"].nunique(),
        concentration["clv_rank_decile"].nunique() == 10,
        "CLV concentration table should have ten value deciles.",
    )

    add_check(
        "threshold_table_has_four_rows",
        len(thresholds),
        len(thresholds) == 4,
        "Threshold table should contain top 1%, 5%, 10%, and 20%.",
    )

    add_check(
        "scenario_table_has_three_rows",
        len(scenarios),
        len(scenarios) == 3,
        "Scenario table should contain conservative, base, and aggressive recovery assumptions.",
    )

    base_recovery = scenarios.loc[
        scenarios["scenario_name"] == "Base recovery",
        "recoverable_portfolio_clv_proxy",
    ].iloc[0]

    add_check(
        "base_recovery_opportunity_positive",
        round(float(base_recovery), 2),
        float(base_recovery) > 0,
        "Base recovery opportunity should be positive.",
    )

    add_check(
        "budget_opportunity_not_empty",
        len(budget_opportunity),
        len(budget_opportunity) > 0,
        "Budget opportunity table should contain retention budget tiers.",
    )

    add_check(
        "model_priority_segments_not_empty",
        len(model_priority),
        len(model_priority) > 0,
        "Model-priority table should contain segment recommendations.",
    )

    report = pd.DataFrame(checks)
    report.to_csv(OUTPUT_DIR / "_revenue_leakage_validation_report.csv", index=False)

    with open(OUTPUT_DIR / "_revenue_leakage_validation_report.json", "w") as f:
        json.dump(checks, f, indent=2)

    print("\nValidation report:")
    print(report.to_string(index=False))

    failed = report[report["status"] == "FAIL"]
    if len(failed) > 0:
        raise ValueError("Revenue leakage validation failed. Review outputs before moving forward.")

    return report


def write_executive_summary(results: dict[str, pd.DataFrame]) -> None:
    portfolio = results["00_revenue_leakage_portfolio_summary"].iloc[0]
    action_groups = results["02_value_leakage_by_action_group"]
    concentration = results["04_clv_concentration_deciles"]
    drivers = results["05_revenue_leakage_driver_scores"]
    budget_pressure = results["06_retention_budget_pressure"]
    thresholds = results["09_value_concentration_thresholds"]
    scenarios = results["10_recoverable_value_scenarios"]
    budget_opportunity = results["11_retention_budget_opportunity"]
    model_priority = results["12_model_priority_segments"]

    top_action_group = action_groups.sort_values(
        "future_churned_clv_proxy", ascending=False
    ).iloc[0]

    top_driver = drivers.sort_values(
        "leakage_priority_score", ascending=False
    ).iloc[0]

    top_budget_pressure = budget_pressure.sort_values(
        "future_churned_clv_proxy", ascending=False
    ).iloc[0]

    top_decile = concentration[concentration["clv_rank_decile"] == 1].iloc[0]

    top_10 = thresholds[thresholds["threshold_label"] == "Top 10%"].iloc[0]
    top_5 = thresholds[thresholds["threshold_label"] == "Top 5%"].iloc[0]

    base_recovery = scenarios[scenarios["scenario_name"] == "Base recovery"].iloc[0]
    best_budget = budget_opportunity.sort_values(
        "net_recovery_opportunity_proxy", ascending=False
    ).iloc[0]
    top_model_segment = model_priority.sort_values(
        "model_priority_score", ascending=False
    ).iloc[0]

    summary = {
        "customers": int(portfolio["customers"]),
        "future_churn_rate": float(portfolio["future_churn_rate"]),
        "annual_revenue_run_rate_proxy": float(portfolio["annual_revenue_run_rate_proxy"]),
        "annual_margin_run_rate_proxy": float(portfolio["annual_margin_run_rate_proxy"]),
        "profit_adjusted_clv_proxy": float(portfolio["profit_adjusted_clv_proxy"]),
        "future_churned_clv_proxy": float(portfolio["future_churned_clv_proxy"]),
        "clv_leakage_rate": float(portfolio["clv_leakage_rate"]),
        "top_action_group": {
            "value_based_action_group": str(top_action_group["value_based_action_group"]),
            "retention_budget_tier": str(top_action_group["retention_budget_tier"]),
            "customers": int(top_action_group["customers"]),
            "future_churn_rate": float(top_action_group["future_churn_rate"]),
            "future_churned_clv_proxy": float(top_action_group["future_churned_clv_proxy"]),
        },
        "top_leakage_driver_segment": {
            "lifecycle_stage": str(top_driver["lifecycle_stage"]),
            "engagement_tier": str(top_driver["engagement_tier"]),
            "revenue_tier": str(top_driver["revenue_tier"]),
            "clv_value_tier": str(top_driver["clv_value_tier"]),
            "action_group": str(top_driver["value_based_action_group"]),
            "customers": int(top_driver["customers"]),
            "future_churn_rate": float(top_driver["future_churn_rate"]),
            "future_churned_clv_proxy": float(top_driver["future_churned_clv_proxy"]),
            "strategy_readout": str(top_driver["leakage_strategy_readout"]),
        },
        "top_value_decile": {
            "customers": int(top_decile["customers"]),
            "clv_share": float(top_decile["clv_share"]),
            "future_churned_clv_share": float(top_decile["future_churned_clv_share"]),
            "future_churn_rate": float(top_decile["future_churn_rate"]),
        },
        "top_5_threshold": {
            "customers": int(top_5["customers"]),
            "clv_share": float(top_5["clv_share"]),
            "future_churned_clv_share": float(top_5["future_churned_clv_share"]),
        },
        "top_10_threshold": {
            "customers": int(top_10["customers"]),
            "clv_share": float(top_10["clv_share"]),
            "future_churned_clv_share": float(top_10["future_churned_clv_share"]),
        },
        "base_recovery_scenario": {
            "recovery_rate": float(base_recovery["recovery_rate"]),
            "recoverable_portfolio_clv_proxy": float(base_recovery["recoverable_portfolio_clv_proxy"]),
            "recoverable_premium_budget_clv_proxy": float(base_recovery["recoverable_premium_budget_clv_proxy"]),
        },
        "best_budget_opportunity": {
            "retention_budget_tier": str(best_budget["retention_budget_tier"]),
            "campaign_cost_proxy": float(best_budget["campaign_cost_proxy"]),
            "recoverable_clv_proxy": float(best_budget["recoverable_clv_proxy"]),
            "net_recovery_opportunity_proxy": float(best_budget["net_recovery_opportunity_proxy"]),
            "roi_proxy": None if pd.isna(best_budget["roi_proxy"]) else float(best_budget["roi_proxy"]),
        },
        "top_model_priority_segment": {
            "lifecycle_stage": str(top_model_segment["lifecycle_stage"]),
            "engagement_tier": str(top_model_segment["engagement_tier"]),
            "revenue_tier": str(top_model_segment["revenue_tier"]),
            "clv_value_tier": str(top_model_segment["clv_value_tier"]),
            "customers": int(top_model_segment["customers"]),
            "future_churned_clv_proxy": float(top_model_segment["future_churned_clv_proxy"]),
            "recommendation": str(top_model_segment["model_priority_recommendation"]),
        },
    }

    with open(OUTPUT_DIR / "_revenue_leakage_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Revenue Leakage and Value Concentration Summary",
        "",
        f"Customers analyzed: {summary['customers']:,}",
        f"Future churn rate in labeled snapshot: {summary['future_churn_rate']:.2%}",
        f"Annual revenue run-rate proxy: {summary['annual_revenue_run_rate_proxy']:,.0f}",
        f"Annual margin run-rate proxy: {summary['annual_margin_run_rate_proxy']:,.0f}",
        f"Profit-adjusted CLV proxy: {summary['profit_adjusted_clv_proxy']:,.0f}",
        f"Future churned CLV proxy: {summary['future_churned_clv_proxy']:,.0f}",
        f"CLV leakage rate: {summary['clv_leakage_rate']:.2%}",
        "",
        "Key readout:",
        (
            f"- The largest value-loss action group is **{summary['top_action_group']['value_based_action_group']}**, "
            f"with {summary['top_action_group']['customers']:,} customers and "
            f"{summary['top_action_group']['future_churned_clv_proxy']:,.0f} future churned CLV proxy."
        ),
        (
            f"- The top leakage driver segment is **{summary['top_leakage_driver_segment']['lifecycle_stage']} / "
            f"{summary['top_leakage_driver_segment']['engagement_tier']} / "
            f"{summary['top_leakage_driver_segment']['revenue_tier']} / "
            f"{summary['top_leakage_driver_segment']['clv_value_tier']}**, with "
            f"{summary['top_leakage_driver_segment']['customers']:,} customers and "
            f"{summary['top_leakage_driver_segment']['future_churned_clv_proxy']:,.0f} future churned CLV proxy."
        ),
        (
            f"- The top 5% of customers represent {summary['top_5_threshold']['clv_share']:.2%} of CLV "
            f"and {summary['top_5_threshold']['future_churned_clv_share']:.2%} of future churned CLV proxy."
        ),
        (
            f"- The top 10% of customers represent {summary['top_10_threshold']['clv_share']:.2%} of CLV "
            f"and {summary['top_10_threshold']['future_churned_clv_share']:.2%} of future churned CLV proxy."
        ),
        (
            f"- At a {summary['base_recovery_scenario']['recovery_rate']:.0%} base recovery assumption, "
            f"recoverable portfolio CLV opportunity is "
            f"{summary['base_recovery_scenario']['recoverable_portfolio_clv_proxy']:,.0f}."
        ),
        (
            f"- The strongest budget opportunity is **{summary['best_budget_opportunity']['retention_budget_tier']}**, "
            f"with estimated net recovery opportunity of "
            f"{summary['best_budget_opportunity']['net_recovery_opportunity_proxy']:,.0f}."
        ),
        "",
        "Business interpretation:",
        (
            "The leakage is not evenly distributed across the customer base. High-value customers create "
            "a disproportionate share of exposed value, which means a generic churn campaign would waste "
            "money. The next model should not optimize for churn probability alone; it should identify "
            "where churn risk, customer value, and intervention economics overlap."
        ),
        "",
        "Modeling implication:",
        (
            f"The top model-priority segment is **{summary['top_model_priority_segment']['lifecycle_stage']} / "
            f"{summary['top_model_priority_segment']['engagement_tier']} / "
            f"{summary['top_model_priority_segment']['revenue_tier']} / "
            f"{summary['top_model_priority_segment']['clv_value_tier']}**. "
            "The churn model should help separate customers who need intervention from customers who are "
            "high value but likely to stay without a paid offer."
        ),
        "",
        "Assumption note:",
        (
            "This script uses the future churn label for retrospective leakage diagnosis and opportunity sizing. "
            "It does not create the final targeting list. Final targeting is created later using predicted churn probability."
        ),
    ]

    summary_path = OUTPUT_DIR / "_executive_revenue_leakage_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved {summary_path}")
    print(f"Saved {OUTPUT_DIR / '_revenue_leakage_summary.json'}")


def main() -> None:
    print("\nRunning revenue leakage and value concentration analysis...")

    if not CLV_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CLV_PATH}. Run src/03_clv_profitability_model.py first."
        )

    if not CUSTOMER_MONTH_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CUSTOMER_MONTH_PATH}. Run src/01_build_customer_month_table.py first."
        )

    with open(OUTPUT_DIR / "_revenue_leakage_assumptions.json", "w") as f:
        json.dump(ASSUMPTIONS, f, indent=2)

    con = duckdb.connect(database=":memory:")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE customer_clv AS
        SELECT * FROM read_parquet('{CLV_PATH}')
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE customer_month AS
        SELECT * FROM read_parquet('{CUSTOMER_MONTH_PATH}')
        """
    )

    write_sql_file()

    results = {}
    for name, query in SQL_QUERIES.items():
        results[name] = export_query(con, name, query)

    validate_outputs(results)
    write_executive_summary(results)

    print("\n04_revenue_leakage_analysis.py complete.")


if __name__ == "__main__":
    main()
