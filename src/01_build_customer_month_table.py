"""
01_build_customer_month_table.py

Customer-month modeling layer for the Customer Revenue Recovery & Retention ROI Engine.

This script turns the cleaned KKBox member, transaction, activity-log, and churn-label
tables into a customer-month analytical model. The customer-month table is the foundation
for cohort retention, churn modeling, CLV, save-worthiness scoring, ROI simulation,
Tableau reporting, and the Streamlit strategy simulator.

The key idea is to move away from a flat churn dataset and create a warehouse-style
customer-month layer: one row per user per observed month, with subscription behavior,
engagement, revenue proxy, lifecycle features, and churn outcome.
"""

from pathlib import Path
import json
import duckdb
import pandas as pd


INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")
SQL_DIR = Path("sql")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SQL_DIR.mkdir(parents=True, exist_ok=True)


CUSTOMER_MONTH_SQL = """
CREATE OR REPLACE TABLE customer_month_table AS
WITH transaction_monthly AS (
    SELECT
        msno,
        transaction_month AS snapshot_month,
        COUNT(*) AS transaction_count,
        SUM(actual_amount_paid) AS monthly_revenue,
        SUM(plan_list_price) AS monthly_list_price,
        SUM(discount_amount) AS monthly_discount_amount,
        AVG(payment_plan_days) AS avg_plan_days,
        MAX(payment_plan_days) AS max_plan_days,
        AVG(paid_per_day) AS avg_paid_per_day,
        MAX(is_auto_renew) AS is_auto_renew,
        MAX(is_cancel) AS had_cancel,
        MAX(is_discounted) AS had_discount,
        MAX(transaction_date_dt) AS latest_transaction_date,
        MAX(membership_expire_date_dt) AS latest_membership_expire_date
    FROM transactions_clean
    GROUP BY
        msno,
        transaction_month
),

activity_monthly AS (
    SELECT
        msno,
        activity_month AS snapshot_month,
        active_days,
        num_25,
        num_50,
        num_75,
        num_985,
        num_100,
        num_unq,
        total_secs,
        total_plays,
        completion_rate_proxy,
        engagement_score
    FROM user_logs_monthly
),

customer_month_keys AS (
    SELECT msno, snapshot_month FROM transaction_monthly
    UNION
    SELECT msno, snapshot_month FROM activity_monthly
),

customer_month_base AS (
    SELECT
        k.msno,
        k.snapshot_month,
        CAST(k.snapshot_month || '-01' AS DATE) AS snapshot_month_date,

        m.city,
        m.age,
        m.gender,
        m.registered_via,
        m.registration_date,

        DATE_DIFF('month', m.registration_date, CAST(k.snapshot_month || '-01' AS DATE)) AS tenure_months,

        COALESCE(t.transaction_count, 0) AS transaction_count,
        COALESCE(t.monthly_revenue, 0) AS monthly_revenue,
        COALESCE(t.monthly_list_price, 0) AS monthly_list_price,
        COALESCE(t.monthly_discount_amount, 0) AS monthly_discount_amount,
        COALESCE(t.avg_plan_days, 0) AS avg_plan_days,
        COALESCE(t.max_plan_days, 0) AS max_plan_days,
        COALESCE(t.avg_paid_per_day, 0) AS avg_paid_per_day,
        COALESCE(t.is_auto_renew, 0) AS is_auto_renew,
        COALESCE(t.had_cancel, 0) AS had_cancel,
        COALESCE(t.had_discount, 0) AS had_discount,
        t.latest_transaction_date,
        t.latest_membership_expire_date,

        COALESCE(a.active_days, 0) AS active_days,
        COALESCE(a.num_25, 0) AS num_25,
        COALESCE(a.num_50, 0) AS num_50,
        COALESCE(a.num_75, 0) AS num_75,
        COALESCE(a.num_985, 0) AS num_985,
        COALESCE(a.num_100, 0) AS num_100,
        COALESCE(a.num_unq, 0) AS num_unq,
        COALESCE(a.total_secs, 0) AS total_secs,
        COALESCE(a.total_plays, 0) AS total_plays,
        COALESCE(a.completion_rate_proxy, 0) AS completion_rate_proxy,
        COALESCE(a.engagement_score, 0) AS engagement_score,

        tr.is_churn AS churn_next_period
    FROM customer_month_keys k
    LEFT JOIN transaction_monthly t
        ON k.msno = t.msno
        AND k.snapshot_month = t.snapshot_month
    LEFT JOIN activity_monthly a
        ON k.msno = a.msno
        AND k.snapshot_month = a.snapshot_month
    LEFT JOIN members_clean m
        ON k.msno = m.msno
    LEFT JOIN train_clean tr
        ON k.msno = tr.msno
),

customer_month_features AS (
    SELECT
        *,
        LAG(engagement_score) OVER (
            PARTITION BY msno
            ORDER BY snapshot_month_date
        ) AS prior_month_engagement_score,

        LAG(total_secs) OVER (
            PARTITION BY msno
            ORDER BY snapshot_month_date
        ) AS prior_month_total_secs,

        LAG(monthly_revenue) OVER (
            PARTITION BY msno
            ORDER BY snapshot_month_date
        ) AS prior_month_revenue,

        AVG(engagement_score) OVER (
            PARTITION BY msno
            ORDER BY snapshot_month_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS trailing_3mo_engagement_score,

        SUM(monthly_revenue) OVER (
            PARTITION BY msno
            ORDER BY snapshot_month_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS trailing_3mo_revenue,

        SUM(active_days) OVER (
            PARTITION BY msno
            ORDER BY snapshot_month_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS trailing_3mo_active_days
    FROM customer_month_base
)

SELECT
    *,
    CASE
        WHEN prior_month_engagement_score IS NULL OR prior_month_engagement_score = 0 THEN NULL
        ELSE (engagement_score - prior_month_engagement_score) / prior_month_engagement_score
    END AS engagement_change_pct,

    CASE
        WHEN prior_month_total_secs IS NULL OR prior_month_total_secs = 0 THEN NULL
        ELSE (total_secs - prior_month_total_secs) / prior_month_total_secs
    END AS activity_change_pct,

    CASE
        WHEN prior_month_revenue IS NULL OR prior_month_revenue = 0 THEN NULL
        ELSE (monthly_revenue - prior_month_revenue) / prior_month_revenue
    END AS revenue_change_pct,

    CASE
        WHEN engagement_score = 0 THEN 'No observed activity'
        WHEN engagement_score < 5 THEN 'Low engagement'
        WHEN engagement_score < 9 THEN 'Medium engagement'
        ELSE 'High engagement'
    END AS engagement_tier,

    CASE
        WHEN monthly_revenue = 0 THEN 'No observed revenue'
        WHEN monthly_revenue < 100 THEN 'Low revenue'
        WHEN monthly_revenue < 180 THEN 'Medium revenue'
        ELSE 'High revenue'
    END AS revenue_tier,

    CASE
        WHEN tenure_months IS NULL THEN 'Unknown tenure'
        WHEN tenure_months < 1 THEN 'New customer'
        WHEN tenure_months < 6 THEN 'Early lifecycle'
        WHEN tenure_months < 24 THEN 'Established'
        ELSE 'Long-tenure'
    END AS lifecycle_stage
FROM customer_month_features
"""


MODELING_SNAPSHOT_SQL = """
CREATE OR REPLACE TABLE modeling_customer_snapshot AS
WITH latest_customer_month AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY msno
            ORDER BY snapshot_month_date DESC
        ) AS rn
    FROM customer_month_table
),

latest_transaction AS (
    SELECT
        msno,
        payment_method_id,
        payment_plan_days AS latest_payment_plan_days,
        plan_list_price AS latest_plan_list_price,
        actual_amount_paid AS latest_actual_amount_paid,
        is_auto_renew AS latest_is_auto_renew,
        is_cancel AS latest_is_cancel,
        transaction_date_dt AS latest_transaction_date_raw,
        membership_expire_date_dt AS latest_membership_expire_date_raw,
        ROW_NUMBER() OVER (
            PARTITION BY msno
            ORDER BY transaction_date_dt DESC, membership_expire_date_dt DESC
        ) AS rn
    FROM transactions_clean
)

SELECT
    tr.msno,
    tr.is_churn AS churn_next_period,

    COALESCE(cm.snapshot_month, 'no_observed_month') AS snapshot_month,
    cm.snapshot_month_date,

    m.city,
    m.age,
    m.gender,
    m.registered_via,
    m.registration_date,

    COALESCE(cm.tenure_months, DATE_DIFF('month', m.registration_date, DATE '2017-03-01')) AS tenure_months,

    COALESCE(cm.transaction_count, 0) AS transaction_count,
    COALESCE(cm.monthly_revenue, 0) AS monthly_revenue,
    COALESCE(cm.monthly_list_price, 0) AS monthly_list_price,
    COALESCE(cm.monthly_discount_amount, 0) AS monthly_discount_amount,
    COALESCE(cm.avg_plan_days, 0) AS avg_plan_days,
    COALESCE(cm.max_plan_days, 0) AS max_plan_days,
    COALESCE(cm.avg_paid_per_day, 0) AS avg_paid_per_day,
    COALESCE(cm.is_auto_renew, 0) AS is_auto_renew,
    COALESCE(cm.had_cancel, 0) AS had_cancel,
    COALESCE(cm.had_discount, 0) AS had_discount,

    COALESCE(cm.active_days, 0) AS active_days,
    COALESCE(cm.num_25, 0) AS num_25,
    COALESCE(cm.num_50, 0) AS num_50,
    COALESCE(cm.num_75, 0) AS num_75,
    COALESCE(cm.num_985, 0) AS num_985,
    COALESCE(cm.num_100, 0) AS num_100,
    COALESCE(cm.num_unq, 0) AS num_unq,
    COALESCE(cm.total_secs, 0) AS total_secs,
    COALESCE(cm.total_plays, 0) AS total_plays,
    COALESCE(cm.completion_rate_proxy, 0) AS completion_rate_proxy,
    COALESCE(cm.engagement_score, 0) AS engagement_score,

    COALESCE(cm.prior_month_engagement_score, 0) AS prior_month_engagement_score,
    COALESCE(cm.prior_month_total_secs, 0) AS prior_month_total_secs,
    COALESCE(cm.prior_month_revenue, 0) AS prior_month_revenue,
    COALESCE(cm.trailing_3mo_engagement_score, 0) AS trailing_3mo_engagement_score,
    COALESCE(cm.trailing_3mo_revenue, 0) AS trailing_3mo_revenue,
    COALESCE(cm.trailing_3mo_active_days, 0) AS trailing_3mo_active_days,
    COALESCE(cm.engagement_change_pct, 0) AS engagement_change_pct,
    COALESCE(cm.activity_change_pct, 0) AS activity_change_pct,
    COALESCE(cm.revenue_change_pct, 0) AS revenue_change_pct,

    COALESCE(cm.engagement_tier, 'No observed activity') AS engagement_tier,
    COALESCE(cm.revenue_tier, 'No observed revenue') AS revenue_tier,
    COALESCE(cm.lifecycle_stage, 'Unknown tenure') AS lifecycle_stage,

    lt.payment_method_id,
    COALESCE(lt.latest_payment_plan_days, 0) AS latest_payment_plan_days,
    COALESCE(lt.latest_plan_list_price, 0) AS latest_plan_list_price,
    COALESCE(lt.latest_actual_amount_paid, 0) AS latest_actual_amount_paid,
    COALESCE(lt.latest_is_auto_renew, 0) AS latest_is_auto_renew,
    COALESCE(lt.latest_is_cancel, 0) AS latest_is_cancel,
    lt.latest_transaction_date_raw,
    lt.latest_membership_expire_date_raw,

    CASE
        WHEN COALESCE(cm.engagement_score, 0) = 0 THEN 1
        ELSE 0
    END AS no_recent_activity_flag,

    CASE
        WHEN COALESCE(cm.had_cancel, 0) = 1 OR COALESCE(lt.latest_is_cancel, 0) = 1 THEN 1
        ELSE 0
    END AS cancellation_signal_flag,

    CASE
        WHEN COALESCE(cm.activity_change_pct, 0) <= -0.5 THEN 1
        ELSE 0
    END AS major_activity_drop_flag

FROM train_clean tr
LEFT JOIN latest_customer_month cm
    ON tr.msno = cm.msno
    AND cm.rn = 1
LEFT JOIN members_clean m
    ON tr.msno = m.msno
LEFT JOIN latest_transaction lt
    ON tr.msno = lt.msno
    AND lt.rn = 1
"""


def main() -> None:
    print("\nBuilding customer-month analytical model...")

    con = duckdb.connect(database=":memory:")

    con.execute(f"""
        CREATE OR REPLACE TABLE train_clean AS
        SELECT * FROM read_parquet('{INTERIM_DIR / "train_clean.parquet"}')
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE members_clean AS
        SELECT * FROM read_parquet('{INTERIM_DIR / "members_clean.parquet"}')
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE transactions_clean AS
        SELECT * FROM read_parquet('{INTERIM_DIR / "transactions_clean.parquet"}')
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE user_logs_monthly AS
        SELECT * FROM read_parquet('{INTERIM_DIR / "user_logs_monthly.parquet"}')
    """)

    with open(SQL_DIR / "01_customer_months.sql", "w") as f:
        f.write(CUSTOMER_MONTH_SQL.strip() + "\n\n" + MODELING_SNAPSHOT_SQL.strip() + "\n")

    con.execute(CUSTOMER_MONTH_SQL)
    con.execute(MODELING_SNAPSHOT_SQL)

    customer_month_path = PROCESSED_DIR / "customer_month_table.parquet"
    modeling_snapshot_path = PROCESSED_DIR / "modeling_customer_snapshot.parquet"
    tableau_base_path = PROCESSED_DIR / "tableau_customer_retention_base.csv"
    summary_path = PROCESSED_DIR / "customer_month_summary.csv"

    con.execute(f"COPY customer_month_table TO '{customer_month_path}' (FORMAT PARQUET)")
    con.execute(f"COPY modeling_customer_snapshot TO '{modeling_snapshot_path}' (FORMAT PARQUET)")

    con.execute(f"""
        COPY (
            SELECT
                *
            FROM modeling_customer_snapshot
        )
        TO '{tableau_base_path}' (HEADER, DELIMITER ',')
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE customer_month_summary AS
        SELECT
            snapshot_month,
            COUNT(*) AS customer_month_rows,
            COUNT(DISTINCT msno) AS unique_customers,
            AVG(churn_next_period) AS churn_rate,
            SUM(monthly_revenue) AS total_monthly_revenue,
            AVG(monthly_revenue) AS avg_monthly_revenue,
            AVG(engagement_score) AS avg_engagement_score,
            AVG(active_days) AS avg_active_days
        FROM customer_month_table
        GROUP BY snapshot_month
        ORDER BY snapshot_month
    """)

    con.execute(f"COPY customer_month_summary TO '{summary_path}' (HEADER, DELIMITER ',')")

    summary = {
        "customer_month_rows": con.execute("SELECT COUNT(*) FROM customer_month_table").fetchone()[0],
        "customer_month_unique_users": con.execute("SELECT COUNT(DISTINCT msno) FROM customer_month_table").fetchone()[0],
        "modeling_snapshot_rows": con.execute("SELECT COUNT(*) FROM modeling_customer_snapshot").fetchone()[0],
        "modeling_snapshot_churn_rate": con.execute("SELECT AVG(churn_next_period) FROM modeling_customer_snapshot").fetchone()[0],
        "users_without_observed_month": con.execute(
            "SELECT COUNT(*) FROM modeling_customer_snapshot WHERE snapshot_month = 'no_observed_month'"
        ).fetchone()[0],
        "customer_month_min_month": con.execute("SELECT MIN(snapshot_month) FROM customer_month_table").fetchone()[0],
        "customer_month_max_month": con.execute("SELECT MAX(snapshot_month) FROM customer_month_table").fetchone()[0],
    }

    with open(PROCESSED_DIR / "customer_month_build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(PROCESSED_DIR / "customer_month_build_summary.txt", "w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print("\nBuild summary:")
    print(json.dumps(summary, indent=2))

    print("\nSaved outputs:")
    print(customer_month_path)
    print(modeling_snapshot_path)
    print(tableau_base_path)
    print(summary_path)
    print(SQL_DIR / "01_customer_months.sql")

    print("\n01_build_customer_month_table.py complete.")


if __name__ == "__main__":
    main()
