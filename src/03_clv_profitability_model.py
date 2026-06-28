"""
03_clv_profitability_model.py

Profit-adjusted customer value layer for the Customer Revenue Recovery &
Retention ROI Engine.

This script estimates customer value before the churn model is built. I keep this
separate on purpose: churn risk answers "who might leave," while CLV answers
"who is worth saving."

The CLV here is a business proxy, not a finance-grade valuation model. It uses
observed revenue, engagement, tenure, auto-renew behavior, cancellation signals,
and a gross margin assumption to create a profit-adjusted customer value score.

Later scripts will combine this value layer with predicted churn risk to create
save-worthiness and retention ROI recommendations.
"""

from pathlib import Path
import json
import duckdb
import pandas as pd


PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = PROCESSED_DIR / "clv_outputs"
SQL_DIR = Path("sql")

MODELING_SNAPSHOT_PATH = PROCESSED_DIR / "modeling_customer_snapshot.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SQL_DIR.mkdir(parents=True, exist_ok=True)

# These are business planning assumptions, not fitted model parameters.
# Keeping them in one dictionary makes the CLV logic easier to audit and change
# before the churn, save-worthiness, and ROI layers use the value score.
ASSUMPTIONS = {
    "gross_margin_rate": 0.65,
    "max_clv_months": 24,
    "base_expected_months": 3,
    "auto_renew_month_bonus": 4,
    "long_tenure_month_bonus": 2,
    "established_tenure_month_bonus": 1,
    "cancellation_month_penalty": -5,
    "major_activity_drop_month_penalty": -2,
    "engagement_decile_weight": 0.55,
    "notes": (
        "This is a pre-model CLV proxy. It does not use the individual churn label "
        "to calculate customer value. The churn label is used only for retrospective "
        "segment summaries showing where future churned value was concentrated."
    ),
}

# This SQL builds one customer-level value row per account.
# The churn label is carried through for retrospective value-at-risk summaries,
# but it is not used to calculate profit-adjusted CLV.
CUSTOMER_CLV_SQL = f"""
CREATE OR REPLACE TABLE customer_clv_scores AS
WITH base AS (
    SELECT
        ms.*,

        -- Use the strongest observed revenue signal as the monthly value baseline.
        -- This avoids undervaluing customers whose latest payment is more informative
        -- than a simple monthly average.
        GREATEST(
            COALESCE(ms.monthly_revenue, 0),
            COALESCE(ms.trailing_3mo_revenue, 0) / 3.0,
            COALESCE(ms.latest_actual_amount_paid, 0)
        ) AS monthly_value_baseline,

        GREATEST(
            COALESCE(ms.engagement_score, 0),
            COALESCE(ms.trailing_3mo_engagement_score, 0)
        ) AS engagement_value_baseline,

        GREATEST(
            COALESCE(ms.is_auto_renew, 0),
            COALESCE(ms.latest_is_auto_renew, 0)
        ) AS auto_renew_signal,

        GREATEST(
            COALESCE(ms.had_cancel, 0),
            COALESCE(ms.latest_is_cancel, 0),
            COALESCE(ms.cancellation_signal_flag, 0)
        ) AS cancellation_signal
    FROM modeling_snapshot ms
),

ranked AS (
    SELECT
        *,
        NTILE(10) OVER (ORDER BY monthly_value_baseline) AS monthly_value_decile,
        NTILE(10) OVER (ORDER BY engagement_value_baseline) AS engagement_decile
    FROM base
),

expected_months AS (
    SELECT
        *,
        -- Expected active months is a bounded planning proxy.
        -- Engagement, tenure, and auto-renew increase expected duration;
        -- cancellation and major activity drops reduce it.
        LEAST(
            {ASSUMPTIONS["max_clv_months"]},
            GREATEST(
                1,
                {ASSUMPTIONS["base_expected_months"]}
                + ({ASSUMPTIONS["engagement_decile_weight"]} * engagement_decile)
                + CASE
                    WHEN auto_renew_signal = 1 THEN {ASSUMPTIONS["auto_renew_month_bonus"]}
                    ELSE 0
                  END
                + CASE
                    WHEN COALESCE(tenure_months, 0) >= 24 THEN {ASSUMPTIONS["long_tenure_month_bonus"]}
                    WHEN COALESCE(tenure_months, 0) >= 6 THEN {ASSUMPTIONS["established_tenure_month_bonus"]}
                    ELSE 0
                  END
                + CASE
                    WHEN cancellation_signal = 1 THEN {ASSUMPTIONS["cancellation_month_penalty"]}
                    ELSE 0
                  END
                + CASE
                    WHEN COALESCE(major_activity_drop_flag, 0) = 1 THEN {ASSUMPTIONS["major_activity_drop_month_penalty"]}
                    ELSE 0
                  END
            )
        ) AS expected_active_months_proxy
    FROM ranked
),

clv_scored AS (
    SELECT
        *,

        {ASSUMPTIONS["gross_margin_rate"]} AS gross_margin_rate,

        -- Profit-adjusted CLV applies the gross margin assumption once.
        -- Later save-worthiness scripts should not multiply this by margin again.
        monthly_value_baseline * {ASSUMPTIONS["gross_margin_rate"]} AS monthly_margin_baseline,
        monthly_value_baseline * 12 AS annual_revenue_run_rate_proxy,
        monthly_value_baseline * 12 * {ASSUMPTIONS["gross_margin_rate"]} AS annual_margin_run_rate_proxy,

        monthly_value_baseline
            * {ASSUMPTIONS["gross_margin_rate"]}
            * expected_active_months_proxy
            AS profit_adjusted_clv_proxy,

        -- Future churned CLV is for retrospective business-impact analysis only.
        -- It is not used to create the CLV score itself.
        CASE
            WHEN churn_next_period = 1 THEN
                monthly_value_baseline
                * {ASSUMPTIONS["gross_margin_rate"]}
                * expected_active_months_proxy
            ELSE 0
        END AS future_churned_clv_proxy,

        CASE
            WHEN monthly_value_baseline <= 0 THEN 'No observed value'
            WHEN monthly_value_decile >= 9 THEN 'Elite value'
            WHEN monthly_value_decile >= 7 THEN 'High value'
            WHEN monthly_value_decile >= 4 THEN 'Core value'
            ELSE 'Low value'
        END AS clv_value_tier,

        -- These action groups are value-based planning labels.
        -- Final retention actions are decided later after churn risk and ROI are added.
        CASE
            WHEN monthly_value_baseline <= 0 THEN 'Suppress paid offer'
            WHEN monthly_value_decile >= 9
                AND cancellation_signal = 1 THEN 'Premium win-back'
            WHEN monthly_value_decile >= 9
                AND no_recent_activity_flag = 1 THEN 'High-value reactivation'
            WHEN monthly_value_decile >= 9 THEN 'Protect high-value customer'
            WHEN monthly_value_decile >= 7
                AND major_activity_drop_flag = 1 THEN 'Save intervention candidate'
            WHEN monthly_value_decile >= 7 THEN 'Targeted retention nurture'
            WHEN monthly_value_decile <= 3 THEN 'Low-cost automated retention'
            ELSE 'Standard lifecycle nurture'
        END AS value_based_action_group,

        CASE
            WHEN monthly_value_baseline <= 0 THEN 'No paid budget'
            WHEN monthly_value_decile >= 9 THEN 'Premium save budget'
            WHEN monthly_value_decile >= 7 THEN 'Moderate save budget'
            WHEN monthly_value_decile >= 4 THEN 'Low-cost save budget'
            ELSE 'Automation only'
        END AS retention_budget_tier

    FROM expected_months
)

SELECT
    msno,
    churn_next_period,

    snapshot_month,
    snapshot_month_date,

    city,
    age,
    gender,
    registered_via,
    registration_date,
    tenure_months,
    lifecycle_stage,

    transaction_count,
    monthly_revenue,
    monthly_list_price,
    monthly_discount_amount,
    avg_plan_days,
    max_plan_days,
    avg_paid_per_day,
    is_auto_renew,
    had_cancel,
    had_discount,

    latest_payment_plan_days,
    latest_plan_list_price,
    latest_actual_amount_paid,
    latest_is_auto_renew,
    latest_is_cancel,
    latest_transaction_date_raw,
    latest_membership_expire_date_raw,

    active_days,
    total_secs,
    total_plays,
    num_unq,
    completion_rate_proxy,
    engagement_score,
    trailing_3mo_engagement_score,
    trailing_3mo_revenue,
    trailing_3mo_active_days,
    engagement_change_pct,
    activity_change_pct,
    revenue_change_pct,

    engagement_tier,
    revenue_tier,

    no_recent_activity_flag,
    cancellation_signal_flag,
    major_activity_drop_flag,

    monthly_value_baseline,
    engagement_value_baseline,
    auto_renew_signal,
    cancellation_signal,

    monthly_value_decile,
    engagement_decile,
    gross_margin_rate,
    expected_active_months_proxy,

    monthly_margin_baseline,
    annual_revenue_run_rate_proxy,
    annual_margin_run_rate_proxy,
    profit_adjusted_clv_proxy,
    future_churned_clv_proxy,

    clv_value_tier,
    value_based_action_group,
    retention_budget_tier

FROM clv_scored
"""

# Summary exports translate the customer-level CLV table into portfolio,
# segment, action-group, and Tableau-ready views.
SUMMARY_QUERIES = {
    "00_portfolio_clv_summary": """
        SELECT
            COUNT(*) AS customers,
            AVG(churn_next_period) AS future_churn_rate,
            SUM(monthly_value_baseline) AS total_monthly_value_baseline,
            SUM(monthly_margin_baseline) AS total_monthly_margin_baseline,
            SUM(annual_revenue_run_rate_proxy) AS annual_revenue_run_rate_proxy,
            SUM(annual_margin_run_rate_proxy) AS annual_margin_run_rate_proxy,
            SUM(profit_adjusted_clv_proxy) AS total_profit_adjusted_clv_proxy,
            SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
            AVG(profit_adjusted_clv_proxy) AS avg_profit_adjusted_clv_proxy,
            AVG(expected_active_months_proxy) AS avg_expected_active_months_proxy
        FROM customer_clv_scores
    """,

    "01_clv_value_tier_summary": """
        SELECT
            clv_value_tier,
            COUNT(*) AS customers,
            AVG(churn_next_period) AS future_churn_rate,
            AVG(monthly_value_baseline) AS avg_monthly_value_baseline,
            AVG(monthly_margin_baseline) AS avg_monthly_margin_baseline,
            SUM(profit_adjusted_clv_proxy) AS total_profit_adjusted_clv_proxy,
            AVG(profit_adjusted_clv_proxy) AS avg_profit_adjusted_clv_proxy,
            SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
            AVG(expected_active_months_proxy) AS avg_expected_active_months_proxy,
            AVG(engagement_score) AS avg_engagement_score,
            AVG(auto_renew_signal) AS auto_renew_rate,
            AVG(cancellation_signal) AS cancellation_signal_rate
        FROM customer_clv_scores
        GROUP BY clv_value_tier
        ORDER BY total_profit_adjusted_clv_proxy DESC
    """,

    "02_clv_segment_summary": """
        SELECT
            lifecycle_stage,
            engagement_tier,
            revenue_tier,
            clv_value_tier,
            COUNT(*) AS customers,
            AVG(churn_next_period) AS future_churn_rate,
            AVG(monthly_value_baseline) AS avg_monthly_value_baseline,
            AVG(profit_adjusted_clv_proxy) AS avg_profit_adjusted_clv_proxy,
            SUM(profit_adjusted_clv_proxy) AS total_profit_adjusted_clv_proxy,
            SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
            AVG(engagement_score) AS avg_engagement_score,
            AVG(active_days) AS avg_active_days,
            AVG(auto_renew_signal) AS auto_renew_rate,
            AVG(cancellation_signal) AS cancellation_signal_rate
        FROM customer_clv_scores
        GROUP BY
            lifecycle_stage,
            engagement_tier,
            revenue_tier,
            clv_value_tier
        HAVING COUNT(*) >= 500
        ORDER BY future_churned_clv_proxy DESC
    """,

    "03_value_based_action_group_summary": """
        SELECT
            value_based_action_group,
            retention_budget_tier,
            COUNT(*) AS customers,
            AVG(churn_next_period) AS future_churn_rate,
            AVG(monthly_value_baseline) AS avg_monthly_value_baseline,
            AVG(profit_adjusted_clv_proxy) AS avg_profit_adjusted_clv_proxy,
            SUM(profit_adjusted_clv_proxy) AS total_profit_adjusted_clv_proxy,
            SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
            AVG(engagement_score) AS avg_engagement_score,
            AVG(auto_renew_signal) AS auto_renew_rate,
            AVG(cancellation_signal) AS cancellation_signal_rate
        FROM customer_clv_scores
        GROUP BY
            value_based_action_group,
            retention_budget_tier
        ORDER BY future_churned_clv_proxy DESC
    """,

    "04_high_value_recovery_targets": """
        SELECT
            msno,
            churn_next_period,
            snapshot_month,
            lifecycle_stage,
            engagement_tier,
            revenue_tier,
            clv_value_tier,
            value_based_action_group,
            retention_budget_tier,
            monthly_value_baseline,
            monthly_margin_baseline,
            expected_active_months_proxy,
            profit_adjusted_clv_proxy,
            future_churned_clv_proxy,
            engagement_score,
            active_days,
            auto_renew_signal,
            cancellation_signal,
            no_recent_activity_flag,
            major_activity_drop_flag
        FROM customer_clv_scores
        WHERE clv_value_tier IN ('Elite value', 'High value')
        ORDER BY
            future_churned_clv_proxy DESC,
            profit_adjusted_clv_proxy DESC
        LIMIT 100000
    """,

    "05_tableau_clv_segment_matrix": """
        SELECT
            lifecycle_stage,
            engagement_tier,
            revenue_tier,
            clv_value_tier,
            value_based_action_group,
            retention_budget_tier,
            COUNT(*) AS customers,
            AVG(churn_next_period) AS future_churn_rate,
            SUM(monthly_value_baseline) AS total_monthly_value_baseline,
            SUM(profit_adjusted_clv_proxy) AS total_profit_adjusted_clv_proxy,
            SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
            AVG(engagement_score) AS avg_engagement_score,
            AVG(auto_renew_signal) AS auto_renew_rate,
            AVG(cancellation_signal) AS cancellation_signal_rate
        FROM customer_clv_scores
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
}


def write_sql_file() -> None:
    sql_path = SQL_DIR / "03_clv_profitability_model.sql"
    with open(sql_path, "w") as f:
        f.write("-- customer_clv_scores\n")
        f.write(CUSTOMER_CLV_SQL.strip())
        f.write(";\n\n")

        for name, query in SUMMARY_QUERIES.items():
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


def validate_outputs(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
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

    rows = con.execute("SELECT COUNT(*) FROM customer_clv_scores").fetchone()[0]
    users = con.execute("SELECT COUNT(DISTINCT msno) FROM customer_clv_scores").fetchone()[0]

    duplicate_users = con.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT msno, COUNT(*) AS row_count
            FROM customer_clv_scores
            GROUP BY msno
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    null_clv = con.execute(
        """
        SELECT COUNT(*)
        FROM customer_clv_scores
        WHERE profit_adjusted_clv_proxy IS NULL
        """
    ).fetchone()[0]

    negative_clv = con.execute(
        """
        SELECT COUNT(*)
        FROM customer_clv_scores
        WHERE profit_adjusted_clv_proxy < 0
           OR monthly_value_baseline < 0
           OR monthly_margin_baseline < 0
        """
    ).fetchone()[0]

    missing_tiers = con.execute(
        """
        SELECT COUNT(*)
        FROM customer_clv_scores
        WHERE clv_value_tier IS NULL
           OR value_based_action_group IS NULL
           OR retention_budget_tier IS NULL
        """
    ).fetchone()[0]
    expected_months_out_of_bounds = con.execute(
        f"""
        SELECT COUNT(*)
        FROM customer_clv_scores
        WHERE expected_active_months_proxy < 1
           OR expected_active_months_proxy > {ASSUMPTIONS["max_clv_months"]}
        """
    ).fetchone()[0]

    add_check(
        "clv_table_not_empty",
        rows,
        rows > 0,
        "Customer CLV table should contain rows.",
    )

    add_check(
        "one_row_per_customer",
        f"rows={rows}, unique_users={users}",
        rows == users,
        "CLV layer should have one row per customer.",
    )

    add_check(
        "no_duplicate_customers",
        duplicate_users,
        duplicate_users == 0,
        "There should be no duplicate customer IDs in the CLV output.",
    )

    add_check(
        "no_null_clv_values",
        null_clv,
        null_clv == 0,
        "Profit-adjusted CLV proxy should be populated for every customer.",
    )

    add_check(
        "no_negative_value_metrics",
        negative_clv,
        negative_clv == 0,
        "Revenue, margin, and CLV proxy values should not be negative.",
    )

    add_check(
        "all_customers_have_value_tiers",
        missing_tiers,
        missing_tiers == 0,
        "Every customer should have a value tier, action group, and budget tier.",
    )

    add_check(
        "expected_active_months_within_bounds",
        expected_months_out_of_bounds,
        expected_months_out_of_bounds == 0,
        "Expected active months should stay within the CLV planning bounds.",
    )

    report = pd.DataFrame(checks)
    report.to_csv(OUTPUT_DIR / "_clv_validation_report.csv", index=False)

    with open(OUTPUT_DIR / "_clv_validation_report.json", "w") as f:
        json.dump(checks, f, indent=2)

    print("\nValidation report:")
    print(report.to_string(index=False))

    failed = report[report["status"] == "FAIL"]
    if len(failed) > 0:
        raise ValueError("CLV validation failed. Review _clv_validation_report.csv before moving forward.")

    return report


def write_executive_summary(results: dict[str, pd.DataFrame]) -> None:
    portfolio = results["00_portfolio_clv_summary"].iloc[0]
    value_tiers = results["01_clv_value_tier_summary"]
    action_groups = results["03_value_based_action_group_summary"]
    top_targets = results["04_high_value_recovery_targets"]

    # These readouts are for the project narrative and README.
    # They summarize where value and future churned value concentrate before
    # the churn model and ROI layers are added.
    top_value_tier = value_tiers.iloc[0]
    top_action_group = action_groups.iloc[0]
    top_target = top_targets.iloc[0]

    summary = {
        "customers": int(portfolio["customers"]),
        "future_churn_rate": float(portfolio["future_churn_rate"]),
        "gross_margin_rate": ASSUMPTIONS["gross_margin_rate"],
        "annual_revenue_run_rate_proxy": float(portfolio["annual_revenue_run_rate_proxy"]),
        "annual_margin_run_rate_proxy": float(portfolio["annual_margin_run_rate_proxy"]),
        "total_profit_adjusted_clv_proxy": float(portfolio["total_profit_adjusted_clv_proxy"]),
        "future_churned_clv_proxy": float(portfolio["future_churned_clv_proxy"]),
        "top_value_tier_by_total_clv": str(top_value_tier["clv_value_tier"]),
        "top_value_tier_customers": int(top_value_tier["customers"]),
        "top_value_tier_future_churn_rate": float(top_value_tier["future_churn_rate"]),
        "top_action_group_by_future_churned_clv": str(top_action_group["value_based_action_group"]),
        "top_action_group_customers": int(top_action_group["customers"]),
        "top_action_group_future_churned_clv_proxy": float(top_action_group["future_churned_clv_proxy"]),
        "highest_ranked_target": {
            "msno": str(top_target["msno"]),
            "clv_value_tier": str(top_target["clv_value_tier"]),
            "action_group": str(top_target["value_based_action_group"]),
            "profit_adjusted_clv_proxy": float(top_target["profit_adjusted_clv_proxy"]),
            "future_churned_clv_proxy": float(top_target["future_churned_clv_proxy"]),
        },
    }

    with open(OUTPUT_DIR / "_clv_profitability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# CLV and Profitability Summary",
        "",
        f"Customers scored: {summary['customers']:,}",
        f"Future churn rate in labeled snapshot: {summary['future_churn_rate']:.2%}",
        f"Gross margin assumption: {summary['gross_margin_rate']:.0%}",
        "",
        "Portfolio value proxy:",
        f"- Annual revenue run-rate proxy: {summary['annual_revenue_run_rate_proxy']:,.0f}",
        f"- Annual margin run-rate proxy: {summary['annual_margin_run_rate_proxy']:,.0f}",
        f"- Total profit-adjusted CLV proxy: {summary['total_profit_adjusted_clv_proxy']:,.0f}",
        f"- Future churned CLV proxy: {summary['future_churned_clv_proxy']:,.0f}",
        "",
        "Key readout:",
        (
            f"- The largest value tier by total CLV is **{summary['top_value_tier_by_total_clv']}**, "
            f"with {summary['top_value_tier_customers']:,} customers and "
            f"{summary['top_value_tier_future_churn_rate']:.2%} future churn."
        ),
        (
            f"- The action group with the most future churned CLV proxy is "
            f"**{summary['top_action_group_by_future_churned_clv']}**, representing "
            f"{summary['top_action_group_customers']:,} customers and "
            f"{summary['top_action_group_future_churned_clv_proxy']:,.0f} future churned CLV proxy."
        ),
        "",
        "Business interpretation:",
        (
            "This layer separates customer value from customer churn risk. That matters because "
            "the retention engine should not simply target the highest-risk customers. It should "
            "prioritize customers where the expected saved margin justifies the intervention cost."
        ),
        "",
        "Assumption note:",
        (
            "This is a pre-model CLV proxy based on observed revenue, margin assumption, tenure, "
            "engagement, auto-renew behavior, cancellation signals, and activity decline. The next "
            "stage will build a churn-risk model and combine predicted churn probability with this "
            "value layer."
        ),
    ]

    summary_path = OUTPUT_DIR / "_executive_clv_profitability_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved {summary_path}")
    print(f"Saved {OUTPUT_DIR / '_clv_profitability_summary.json'}")


def main() -> None:
    print("\nBuilding CLV and profitability layer...")

    if not MODELING_SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODELING_SNAPSHOT_PATH}. Run src/01_build_customer_month_table.py first."
        )

    with open(OUTPUT_DIR / "_clv_assumptions.json", "w") as f:
        json.dump(ASSUMPTIONS, f, indent=2)

    con = duckdb.connect(database=":memory:")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE modeling_snapshot AS
        SELECT * FROM read_parquet('{MODELING_SNAPSHOT_PATH}')
        """
    )

    con.execute(CUSTOMER_CLV_SQL)

    write_sql_file()

    clv_parquet_path = OUTPUT_DIR / "customer_clv_scores.parquet"
    tableau_sample_path = OUTPUT_DIR / "tableau_customer_clv_sample.csv"

    con.execute(f"COPY customer_clv_scores TO '{clv_parquet_path}' (FORMAT PARQUET)")

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM customer_clv_scores
            ORDER BY profit_adjusted_clv_proxy DESC
            LIMIT 100000
        )
        TO '{tableau_sample_path}' (HEADER, DELIMITER ',')
        """
    )

    print(f"Saved {clv_parquet_path}")
    print(f"Saved {tableau_sample_path}")

    validate_outputs(con)

    results = {}
    for name, query in SUMMARY_QUERIES.items():
        results[name] = export_query(con, name, query)

    write_executive_summary(results)

    print("\n03_clv_profitability_model.py complete.")


if __name__ == "__main__":
    main()
