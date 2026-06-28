"""
02_sql_cohort_retention_analysis.py

Cohort retention and revenue leakage layer for the Customer Revenue Recovery &
Retention ROI Engine.

This script uses the customer-month table from 01_build_customer_month_table.py
to answer the first business questions in the project:

Where does customer activity weaken?
Which lifecycle, engagement, and revenue groups carry the most future churn risk?
Where is subscription revenue most exposed?

Important modeling note:
KKBox train_v2 provides a future churn label for the labeled users. I do not treat
that label as if churn happened in every historical month. For historical
customer-month views, I call the measure "future churn label" or "future churned
revenue proxy." For modeling and strategy views, I use the latest customer snapshot
so the churn label is aligned to the decision point.

Outputs:
- data/processed/cohort_retention_outputs/*.csv
- data/processed/cohort_retention_outputs/_cohort_retention_summary.json
- data/processed/cohort_retention_outputs/_executive_cohort_retention_summary.md
- sql/02_cohort_retention_analysis.sql

This layer is descriptive. It finds where risk and exposed revenue concentrate
before the later CLV, churn model, save-worthiness, and ROI layers make targeting
decisions.
"""

from pathlib import Path
import json
import duckdb
import pandas as pd


PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = PROCESSED_DIR / "cohort_retention_outputs"
SQL_DIR = Path("sql")

CUSTOMER_MONTH_PATH = PROCESSED_DIR / "customer_month_table.parquet"
MODELING_SNAPSHOT_PATH = PROCESSED_DIR / "modeling_customer_snapshot.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SQL_DIR.mkdir(parents=True, exist_ok=True)


REQUIRED_INPUTS = [CUSTOMER_MONTH_PATH, MODELING_SNAPSHOT_PATH]


def check_required_inputs() -> None:
    missing = [str(path) for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required processed file(s): "
            + ", ".join(missing)
            + ". Run src/01_build_customer_month_table.py first."
        )


# These SQL exports are intentionally descriptive. They diagnose activity,
# cohort, lifecycle, engagement, and revenue exposure before predictive modeling.
SQL_QUERIES = {
    "01_monthly_revenue_leakage": """
        SELECT
            -- This is a historical customer-month view. The churn label is a future
            -- label attached to the customer, not a churn event inside each month.
            snapshot_month,
            COUNT(DISTINCT msno) AS observed_customers,
            SUM(monthly_revenue) AS total_monthly_revenue_proxy,
            AVG(monthly_revenue) AS avg_monthly_revenue_proxy,
            AVG(churn_next_period) AS future_churn_label_rate,
            SUM(CASE WHEN churn_next_period = 1 THEN monthly_revenue ELSE 0 END)
                AS future_churned_revenue_proxy,
            AVG(engagement_score) AS avg_engagement_score,
            AVG(active_days) AS avg_active_days,
            AVG(is_auto_renew) AS auto_renew_rate,
            AVG(had_cancel) AS cancel_signal_rate
        FROM customer_month
        GROUP BY snapshot_month
        ORDER BY snapshot_month
    """,

    "02_cohort_retention_long": """
        WITH customer_first_month AS (
            -- Cohorts are based on first observed activity/payment month in this
            -- project data, not the customer's true lifetime signup month.
            SELECT
                msno,
                MIN(snapshot_month_date) AS first_observed_month_date,
                MIN(snapshot_month) AS first_observed_month
            FROM customer_month
            GROUP BY msno
        ),

        cohort_sizes AS (
            SELECT
                first_observed_month,
                first_observed_month_date,
                COUNT(DISTINCT msno) AS cohort_size
            FROM customer_first_month
            GROUP BY first_observed_month, first_observed_month_date
        ),

        cohort_activity AS (
            SELECT
                f.first_observed_month,
                f.first_observed_month_date,
                DATE_DIFF('month', f.first_observed_month_date, cm.snapshot_month_date)
                    AS months_since_first_observed,
                COUNT(DISTINCT cm.msno) AS observed_users,
                SUM(cm.monthly_revenue) AS cohort_monthly_revenue_proxy,
                SUM(CASE WHEN cm.churn_next_period = 1 THEN cm.monthly_revenue ELSE 0 END)
                    AS future_churned_revenue_proxy,
                AVG(cm.churn_next_period) AS future_churn_label_rate,
                AVG(cm.engagement_score) AS avg_engagement_score
            FROM customer_month cm
            INNER JOIN customer_first_month f
                ON cm.msno = f.msno
            WHERE DATE_DIFF('month', f.first_observed_month_date, cm.snapshot_month_date) BETWEEN 0 AND 12
            GROUP BY
                f.first_observed_month,
                f.first_observed_month_date,
                months_since_first_observed
        )

        SELECT
            a.first_observed_month,
            a.months_since_first_observed,
            s.cohort_size,
            a.observed_users,
            -- Observed retention means the customer appears again in the customer-month table.
            -- It should not be read as exact contractual renewal survival.
            CAST(a.observed_users AS DOUBLE) / NULLIF(s.cohort_size, 0) AS observed_retention_rate,
            a.cohort_monthly_revenue_proxy,
            a.future_churned_revenue_proxy,
            a.future_churn_label_rate,
            a.avg_engagement_score
        FROM cohort_activity a
        INNER JOIN cohort_sizes s
            ON a.first_observed_month = s.first_observed_month
        ORDER BY
            a.first_observed_month,
            a.months_since_first_observed
    """,

    "03_engagement_tier_churn_revenue": """
        SELECT
            engagement_tier,
            COUNT(*) AS customers,
            AVG(churn_next_period) AS churn_rate,
            SUM(monthly_revenue) AS total_monthly_revenue_proxy,
            AVG(monthly_revenue) AS avg_monthly_revenue_proxy,
            SUM(CASE WHEN churn_next_period = 1 THEN monthly_revenue ELSE 0 END)
                AS future_churned_revenue_proxy,
            AVG(engagement_score) AS avg_engagement_score,
            AVG(active_days) AS avg_active_days,
            AVG(no_recent_activity_flag) AS no_recent_activity_rate,
            AVG(major_activity_drop_flag) AS major_activity_drop_rate
        FROM modeling_snapshot
        GROUP BY engagement_tier
        ORDER BY future_churned_revenue_proxy DESC
    """,

    "04_revenue_tier_churn_revenue": """
        SELECT
            revenue_tier,
            COUNT(*) AS customers,
            AVG(churn_next_period) AS churn_rate,
            SUM(monthly_revenue) AS total_monthly_revenue_proxy,
            AVG(monthly_revenue) AS avg_monthly_revenue_proxy,
            SUM(CASE WHEN churn_next_period = 1 THEN monthly_revenue ELSE 0 END)
                AS future_churned_revenue_proxy,
            AVG(engagement_score) AS avg_engagement_score,
            AVG(had_discount) AS discount_exposure_rate,
            AVG(is_auto_renew) AS auto_renew_rate
        FROM modeling_snapshot
        GROUP BY revenue_tier
        ORDER BY future_churned_revenue_proxy DESC
    """,

    "05_lifecycle_stage_churn_revenue": """
        SELECT
            lifecycle_stage,
            COUNT(*) AS customers,
            AVG(churn_next_period) AS churn_rate,
            SUM(monthly_revenue) AS total_monthly_revenue_proxy,
            AVG(monthly_revenue) AS avg_monthly_revenue_proxy,
            SUM(CASE WHEN churn_next_period = 1 THEN monthly_revenue ELSE 0 END)
                AS future_churned_revenue_proxy,
            AVG(tenure_months) AS avg_tenure_months,
            AVG(engagement_score) AS avg_engagement_score,
            AVG(cancellation_signal_flag) AS cancellation_signal_rate
        FROM modeling_snapshot
        GROUP BY lifecycle_stage
        ORDER BY future_churned_revenue_proxy DESC
    """,

    "06_registered_channel_churn_revenue": """
        SELECT
            registered_via,
            COUNT(*) AS customers,
            AVG(churn_next_period) AS churn_rate,
            SUM(monthly_revenue) AS total_monthly_revenue_proxy,
            AVG(monthly_revenue) AS avg_monthly_revenue_proxy,
            SUM(CASE WHEN churn_next_period = 1 THEN monthly_revenue ELSE 0 END)
                AS future_churned_revenue_proxy,
            AVG(engagement_score) AS avg_engagement_score,
            AVG(is_auto_renew) AS auto_renew_rate
        FROM modeling_snapshot
        GROUP BY registered_via
        ORDER BY future_churned_revenue_proxy DESC
    """,

    "07_retention_risk_segment_matrix": """
        SELECT
            -- Segment matrix uses the latest modeling snapshot so each customer
            -- contributes once to the retention strategy readout.
            lifecycle_stage,
            engagement_tier,
            revenue_tier,
            COUNT(*) AS customers,
            AVG(churn_next_period) AS churn_rate,
            SUM(monthly_revenue) AS total_monthly_revenue_proxy,
            AVG(monthly_revenue) AS avg_monthly_revenue_proxy,
            SUM(CASE WHEN churn_next_period = 1 THEN monthly_revenue ELSE 0 END)
                AS future_churned_revenue_proxy,
            AVG(engagement_score) AS avg_engagement_score,
            AVG(active_days) AS avg_active_days,
            AVG(cancellation_signal_flag) AS cancellation_signal_rate,
            AVG(major_activity_drop_flag) AS major_activity_drop_rate
        FROM modeling_snapshot
        GROUP BY
            lifecycle_stage,
            engagement_tier,
            revenue_tier
        HAVING COUNT(*) >= 500
        ORDER BY
            future_churned_revenue_proxy DESC,
            churn_rate DESC
    """,

    "08_high_value_at_risk_segments": """
        WITH segment_base AS (
            SELECT
                lifecycle_stage,
                engagement_tier,
                revenue_tier,
                latest_is_auto_renew,
                latest_is_cancel,
                COUNT(*) AS customers,
                AVG(churn_next_period) AS churn_rate,
                AVG(monthly_revenue) AS avg_monthly_revenue_proxy,
                SUM(monthly_revenue) AS total_monthly_revenue_proxy,
                SUM(CASE WHEN churn_next_period = 1 THEN monthly_revenue ELSE 0 END)
                    AS future_churned_revenue_proxy,
                AVG(engagement_score) AS avg_engagement_score,
                AVG(major_activity_drop_flag) AS major_activity_drop_rate,
                AVG(cancellation_signal_flag) AS cancellation_signal_rate
            FROM modeling_snapshot
            GROUP BY
                lifecycle_stage,
                engagement_tier,
                revenue_tier,
                latest_is_auto_renew,
                latest_is_cancel
            HAVING COUNT(*) >= 500
        )

        SELECT
            *,
            -- This score is a descriptive prioritization heuristic. It identifies
            -- segments worth deeper CLV/model review, not final campaign eligibility.
            future_churned_revenue_proxy
                * (1 + cancellation_signal_rate)
                AS revenue_recovery_priority_score,

            CASE
                WHEN latest_is_cancel = 1
                    AND revenue_tier IN ('High revenue', 'Medium revenue')
                    THEN 'Win-back recovery candidate'
                WHEN revenue_tier = 'High revenue'
                    AND churn_rate >= 0.10
                    AND latest_is_cancel = 0
                    THEN 'High-priority retention candidate'
                WHEN engagement_tier = 'No observed activity'
                    AND churn_rate >= 0.10
                    THEN 'Low-cost reactivation candidate'
                WHEN lifecycle_stage IN ('New customer', 'Early lifecycle')
                    AND engagement_tier IN ('No observed activity', 'Low engagement')
                    THEN 'Onboarding intervention candidate'
                WHEN churn_rate < 0.05
                    AND revenue_tier = 'High revenue'
                    THEN 'Monitor and protect'
                ELSE 'Standard retention monitoring'
            END AS recommended_retention_focus

        FROM segment_base
        ORDER BY revenue_recovery_priority_score DESC
        LIMIT 30
    """,

    "09_tableau_cohort_heatmap": """
        WITH customer_first_month AS (
            SELECT
                msno,
                MIN(snapshot_month_date) AS first_observed_month_date,
                MIN(snapshot_month) AS first_observed_month
            FROM customer_month
            GROUP BY msno
        ),

        cohort_sizes AS (
            SELECT
                first_observed_month,
                COUNT(DISTINCT msno) AS cohort_size
            FROM customer_first_month
            GROUP BY first_observed_month
        ),

        cohort_activity AS (
            SELECT
                f.first_observed_month,
                DATE_DIFF('month', f.first_observed_month_date, cm.snapshot_month_date)
                    AS months_since_first_observed,
                COUNT(DISTINCT cm.msno) AS observed_users
            FROM customer_month cm
            INNER JOIN customer_first_month f
                ON cm.msno = f.msno
            WHERE DATE_DIFF('month', f.first_observed_month_date, cm.snapshot_month_date) BETWEEN 0 AND 12
            GROUP BY
                f.first_observed_month,
                months_since_first_observed
        )

        SELECT
            a.first_observed_month AS cohort_month,
            a.months_since_first_observed,
            s.cohort_size,
            a.observed_users,
            CAST(a.observed_users AS DOUBLE) / NULLIF(s.cohort_size, 0) AS observed_retention_rate
        FROM cohort_activity a
        INNER JOIN cohort_sizes s
            ON a.first_observed_month = s.first_observed_month
        ORDER BY
            cohort_month,
            months_since_first_observed
    """,
}


# Save the SQL separately so reviewers can inspect the business logic without
# opening the Python orchestration code.
def write_sql_file() -> None:
    sql_path = SQL_DIR / "02_cohort_retention_analysis.sql"
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


# Tableau and spreadsheet review are easier with both a long cohort table and
# a wide matrix version.
def create_cohort_matrix(cohort_long: pd.DataFrame) -> pd.DataFrame:
    matrix = cohort_long.pivot_table(
        index="first_observed_month",
        columns="months_since_first_observed",
        values="observed_retention_rate",
        aggfunc="mean",
    ).reset_index()

    matrix.columns = [
        "first_observed_month" if col == "first_observed_month" else f"month_{int(col)}"
        for col in matrix.columns
    ]

    output_path = OUTPUT_DIR / "02b_cohort_retention_matrix.csv"
    matrix.to_csv(output_path, index=False)
    print(f"Saved {output_path} | rows={len(matrix):,} cols={matrix.shape[1]:,}")
    return matrix


# Write a recruiter-readable summary that connects the descriptive cohort layer
# to the later CLV, churn modeling, and ROI decision layers.
def write_executive_summary(results: dict[str, pd.DataFrame]) -> None:
    monthly = results["01_monthly_revenue_leakage"]
    engagement = results["03_engagement_tier_churn_revenue"]
    lifecycle = results["05_lifecycle_stage_churn_revenue"]
    risk_segments = results["08_high_value_at_risk_segments"]

    latest_month = monthly.sort_values("snapshot_month").tail(1).iloc[0]
    highest_engagement_risk = engagement.sort_values(
        "future_churned_revenue_proxy", ascending=False
    ).iloc[0]
    highest_lifecycle_risk = lifecycle.sort_values(
        "future_churned_revenue_proxy", ascending=False
    ).iloc[0]
    top_segment = risk_segments.iloc[0]

    summary = {
        "latest_observed_month": str(latest_month["snapshot_month"]),
        "latest_observed_customers": int(latest_month["observed_customers"]),
        "latest_month_future_churn_label_rate": float(latest_month["future_churn_label_rate"]),
        "latest_month_future_churned_revenue_proxy": float(
            latest_month["future_churned_revenue_proxy"]
        ),
        "highest_future_churned_revenue_engagement_tier": str(highest_engagement_risk["engagement_tier"]),
        "highest_future_churned_revenue_lifecycle_stage": str(highest_lifecycle_risk["lifecycle_stage"]),
        "top_recovery_segment": {
            "lifecycle_stage": str(top_segment["lifecycle_stage"]),
            "engagement_tier": str(top_segment["engagement_tier"]),
            "revenue_tier": str(top_segment["revenue_tier"]),
            "customers": int(top_segment["customers"]),
            "churn_rate": float(top_segment["churn_rate"]),
            "future_churned_revenue_proxy": float(
                top_segment["future_churned_revenue_proxy"]
            ),
            "recommended_focus": str(top_segment["recommended_retention_focus"]),
        },
    }

    with open(OUTPUT_DIR / "_cohort_retention_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Cohort Retention and Revenue Leakage Summary",
        "",
        f"Latest observed month: {summary['latest_observed_month']}",
        f"Observed customers in latest month: {summary['latest_observed_customers']:,}",
        f"Latest-month future churn label rate: {summary['latest_month_future_churn_label_rate']:.2%}",
        f"Latest-month future churned revenue proxy: {summary['latest_month_future_churned_revenue_proxy']:,.0f}",
        "",
        "Key readout:",
        f"- The engagement tier with the most future churned revenue proxy is **{summary['highest_future_churned_revenue_engagement_tier']}**.",
        f"- The lifecycle stage with the most future churned revenue proxy is **{summary['highest_future_churned_revenue_lifecycle_stage']}**.",
        (
            "- The top recovery segment is "
            f"**{summary['top_recovery_segment']['lifecycle_stage']} / "
            f"{summary['top_recovery_segment']['engagement_tier']} / "
            f"{summary['top_recovery_segment']['revenue_tier']}**, with "
            f"{summary['top_recovery_segment']['customers']:,} customers, "
            f"{summary['top_recovery_segment']['churn_rate']:.2%} churn risk, and "
            f"{summary['top_recovery_segment']['future_churned_revenue_proxy']:,.0f} future churned revenue proxy."
        ),
        "",
        "Business interpretation:",
        (
            "This layer identifies where retention pressure is concentrated before any predictive model is used. "
            "The next stage will turn these cohort and segment patterns into profit-adjusted CLV, "
            "churn-risk scoring, and save-worthiness prioritization."
        ),
        "",
        "Assumption note:",
        (
            "The KKBox label is a future churn outcome, so historical customer-month rows should be interpreted "
            "as prior behavior associated with future churn, not as monthly churn events."
        ),
    ]

    summary_path = OUTPUT_DIR / "_executive_cohort_retention_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved {summary_path}")
    print(f"Saved {OUTPUT_DIR / '_cohort_retention_summary.json'}")


def main() -> None:
    print("\nRunning cohort retention and revenue leakage analysis...")

    check_required_inputs()

    con = duckdb.connect(database=":memory:")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE customer_month AS
        SELECT * FROM read_parquet('{CUSTOMER_MONTH_PATH}')
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE modeling_snapshot AS
        SELECT * FROM read_parquet('{MODELING_SNAPSHOT_PATH}')
        """
    )

    write_sql_file()

    results = {}
    for name, query in SQL_QUERIES.items():
        results[name] = export_query(con, name, query)

    create_cohort_matrix(results["02_cohort_retention_long"])
    write_executive_summary(results)

    print("\n02_sql_cohort_retention_analysis.py complete.")


if __name__ == "__main__":
    main()
