"""
06_survival_lifecycle_analysis.py

Lifecycle and survival analysis layer for the Customer Revenue Recovery &
Retention ROI Engine.

The churn model tells me which customers rank high today. This script looks at
the same problem from a lifecycle angle: how long customers stay observable,
where risk concentrates over the customer lifecycle, and which customer groups
deserve intervention timing.

This is not a clinical survival study. The KKBox label is a future churn outcome,
so I treat the final churn flag as the event and the customer's observed months
as the lifecycle duration. The output is a practical lifecycle-risk view for
retention strategy, Tableau, and the later save-worthiness engine.
"""

from pathlib import Path
import json
import duckdb
import pandas as pd


PROCESSED_DIR = Path("data/processed")
CHURN_DIR = PROCESSED_DIR / "churn_model_outputs"
OUTPUT_DIR = PROCESSED_DIR / "survival_lifecycle_outputs"
SQL_DIR = Path("sql")

CUSTOMER_MONTH_PATH = PROCESSED_DIR / "customer_month_table.parquet"
CHURN_SCORED_PATH = CHURN_DIR / "churn_scored_customers.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SQL_DIR.mkdir(parents=True, exist_ok=True)

# Cap lifecycle curves at 36 months so the chart does not over-read sparse
# long-tail customer-month coverage. Smaller segment curves are suppressed
# because lifecycle timing can be noisy in small groups.
MAX_SURVIVAL_MONTH = 36
MIN_GROUP_SIZE = 500


# Build one customer-level lifecycle row by combining scored churn customers with
# observed customer-month coverage. Duration is based on observed coverage, so
# this is a lifecycle-risk proxy rather than a precise event-history dataset.
LIFECYCLE_BASE_SQL = """
CREATE OR REPLACE TABLE customer_lifecycle_survival_base AS
WITH month_bounds AS (
    SELECT
        msno,
        MIN(snapshot_month_date) AS first_observed_month_date,
        MAX(snapshot_month_date) AS last_observed_month_date,
        MIN(snapshot_month) AS first_observed_month,
        MAX(snapshot_month) AS last_observed_month,
        -- Count the months where the customer appears in the customer-month table.
        -- This is observed coverage, not a guarantee of full lifetime history.
        COUNT(DISTINCT snapshot_month) AS observed_active_months
    FROM customer_month
    GROUP BY msno
),

base AS (
    SELECT
        cs.msno,
        cs.actual_future_churn_label AS churn_event,
        cs.predicted_churn_probability,
        cs.churn_risk_decile,
        cs.churn_risk_tier,

        cs.snapshot_month,
        cs.lifecycle_stage,
        cs.engagement_tier,
        cs.revenue_tier,
        cs.clv_value_tier,
        cs.value_based_action_group,
        cs.retention_budget_tier,

        cs.monthly_value_baseline,
        cs.monthly_margin_baseline,
        cs.annual_revenue_run_rate_proxy,
        cs.annual_margin_run_rate_proxy,
        cs.profit_adjusted_clv_proxy,
        cs.future_churned_clv_proxy,

        mb.first_observed_month,
        mb.last_observed_month,
        mb.first_observed_month_date,
        mb.last_observed_month_date,
        COALESCE(mb.observed_active_months, 0) AS observed_active_months,

        -- Duration is measured from first to last observed customer-month.
        -- I do not treat this as an exact cancellation timestamp.
        CASE
            WHEN mb.first_observed_month_date IS NULL OR mb.last_observed_month_date IS NULL THEN 1
            ELSE DATE_DIFF('month', mb.first_observed_month_date, mb.last_observed_month_date) + 1
        END AS observation_span_months

    FROM churn_scored_customers cs
    LEFT JOIN month_bounds mb
        ON cs.msno = mb.msno
)

SELECT
    *,
    CASE
        WHEN observation_span_months <= 0 THEN 1
        ELSE observation_span_months
    END AS lifecycle_duration_months,

    CASE
        WHEN observation_span_months > 0
        THEN observed_active_months / observation_span_months
        ELSE 0
    END AS observed_month_coverage_rate,

    -- Combine churn risk with value because high-risk alone is not enough
    -- to justify paid retention spend.
    CASE
        WHEN churn_risk_tier IN ('Critical risk', 'High risk')
            AND clv_value_tier IN ('Elite value', 'High value')
            THEN 'High risk / high value'
        WHEN churn_risk_tier IN ('Critical risk', 'High risk')
            AND clv_value_tier IN ('Low value', 'No observed value')
            THEN 'High risk / low value'
        WHEN churn_risk_tier IN ('Low risk')
            AND clv_value_tier IN ('Elite value', 'High value')
            THEN 'Low risk / high value'
        WHEN churn_risk_tier IN ('Medium risk')
            AND clv_value_tier IN ('Elite value', 'High value')
            THEN 'Medium risk / high value'
        ELSE 'Standard lifecycle population'
    END AS value_risk_quadrant,

    CASE
        WHEN churn_risk_tier = 'Critical risk'
            AND clv_value_tier IN ('Elite value', 'High value')
            THEN 'Immediate save-worthiness review'
        WHEN churn_risk_tier IN ('Critical risk', 'High risk')
            AND retention_budget_tier = 'Premium save budget'
            THEN 'Premium save timing analysis'
        WHEN churn_risk_tier IN ('Critical risk', 'High risk')
            AND retention_budget_tier IN ('Automation only', 'No paid budget')
            THEN 'Do not overspend'
        WHEN churn_risk_tier = 'Low risk'
            AND clv_value_tier IN ('Elite value', 'High value')
            THEN 'Protect without discount'
        ELSE 'Monitor lifecycle trend'
    END AS lifecycle_strategy_readout

FROM base
"""


# Summary exports turn the customer-level lifecycle base into portfolio,
# timing, value/risk, model-priority, and Tableau-ready views.
SUMMARY_QUERIES = {
    "00_portfolio_lifecycle_summary": """
        SELECT
            COUNT(*) AS customers,
            AVG(churn_event) AS future_churn_rate,
            AVG(predicted_churn_probability) AS avg_predicted_churn_probability,
            AVG(lifecycle_duration_months) AS avg_lifecycle_duration_months,
            MEDIAN(lifecycle_duration_months) AS median_lifecycle_duration_months,
            AVG(observed_active_months) AS avg_observed_active_months,
            AVG(observed_month_coverage_rate) AS avg_observed_month_coverage_rate,
            SUM(profit_adjusted_clv_proxy) AS profit_adjusted_clv_proxy,
            SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
            SUM(CASE WHEN value_risk_quadrant = 'High risk / high value' THEN 1 ELSE 0 END)
                AS high_risk_high_value_customers,
            SUM(CASE WHEN value_risk_quadrant = 'High risk / high value' THEN future_churned_clv_proxy ELSE 0 END)
                AS high_risk_high_value_future_churned_clv_proxy
        FROM customer_lifecycle_survival_base
    """,

    "01_lifecycle_stage_summary": """
        SELECT
            lifecycle_stage,
            COUNT(*) AS customers,
            AVG(churn_event) AS future_churn_rate,
            AVG(predicted_churn_probability) AS avg_predicted_churn_probability,
            AVG(lifecycle_duration_months) AS avg_lifecycle_duration_months,
            MEDIAN(lifecycle_duration_months) AS median_lifecycle_duration_months,
            AVG(observed_month_coverage_rate) AS avg_observed_month_coverage_rate,
            SUM(profit_adjusted_clv_proxy) AS profit_adjusted_clv_proxy,
            SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
            AVG(monthly_value_baseline) AS avg_monthly_value_baseline
        FROM customer_lifecycle_survival_base
        GROUP BY lifecycle_stage
        ORDER BY future_churned_clv_proxy DESC
    """,

    "02_monthly_hazard_by_observed_age": """
        WITH durations AS (
            SELECT
                lifecycle_duration_months,
                churn_event
            FROM customer_lifecycle_survival_base
            WHERE lifecycle_duration_months BETWEEN 1 AND 36
        ),

        months AS (
            SELECT range AS month_index
            FROM range(1, 37)
        )

        SELECT
            m.month_index,
            SUM(CASE WHEN d.lifecycle_duration_months >= m.month_index THEN 1 ELSE 0 END) AS customers_at_risk,
            SUM(CASE WHEN d.lifecycle_duration_months = m.month_index AND d.churn_event = 1 THEN 1 ELSE 0 END) AS churn_events,
            SUM(CASE WHEN d.lifecycle_duration_months = m.month_index AND d.churn_event = 0 THEN 1 ELSE 0 END) AS censored_customers,
            CASE
                WHEN SUM(CASE WHEN d.lifecycle_duration_months >= m.month_index THEN 1 ELSE 0 END) > 0
                THEN
                    SUM(CASE WHEN d.lifecycle_duration_months = m.month_index AND d.churn_event = 1 THEN 1 ELSE 0 END)
                    / SUM(CASE WHEN d.lifecycle_duration_months >= m.month_index THEN 1 ELSE 0 END)
                ELSE 0
            END AS monthly_hazard_rate
        FROM months m
        CROSS JOIN durations d
        GROUP BY m.month_index
        ORDER BY m.month_index
    """,

    "03_value_risk_quadrant_summary": """
        SELECT
            value_risk_quadrant,
            lifecycle_strategy_readout,
            COUNT(*) AS customers,
            AVG(churn_event) AS future_churn_rate,
            AVG(predicted_churn_probability) AS avg_predicted_churn_probability,
            AVG(lifecycle_duration_months) AS avg_lifecycle_duration_months,
            SUM(profit_adjusted_clv_proxy) AS profit_adjusted_clv_proxy,
            SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
            AVG(monthly_value_baseline) AS avg_monthly_value_baseline
        FROM customer_lifecycle_survival_base
        GROUP BY
            value_risk_quadrant,
            lifecycle_strategy_readout
        ORDER BY future_churned_clv_proxy DESC
    """,

    "04_model_priority_lifecycle_segments": """
        WITH segment_base AS (
            SELECT
                lifecycle_stage,
                engagement_tier,
                revenue_tier,
                clv_value_tier,
                churn_risk_tier,
                value_risk_quadrant,
                lifecycle_strategy_readout,
                retention_budget_tier,
                COUNT(*) AS customers,
                AVG(churn_event) AS future_churn_rate,
                AVG(predicted_churn_probability) AS avg_predicted_churn_probability,
                AVG(lifecycle_duration_months) AS avg_lifecycle_duration_months,
                MEDIAN(lifecycle_duration_months) AS median_lifecycle_duration_months,
                AVG(observed_month_coverage_rate) AS avg_observed_month_coverage_rate,
                SUM(profit_adjusted_clv_proxy) AS profit_adjusted_clv_proxy,
                SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
                AVG(monthly_value_baseline) AS avg_monthly_value_baseline
            FROM customer_lifecycle_survival_base
            GROUP BY
                lifecycle_stage,
                engagement_tier,
                revenue_tier,
                clv_value_tier,
                churn_risk_tier,
                value_risk_quadrant,
                lifecycle_strategy_readout,
                retention_budget_tier
            HAVING COUNT(*) >= 500
        )

        SELECT
            *,
            -- Segment priority is a planning score for model attention.
            -- Final offer decisions happen later through save-worthiness and ROI logic.
            future_churned_clv_proxy
                * (1 + avg_predicted_churn_probability)
                * CASE
                    WHEN value_risk_quadrant = 'High risk / high value' THEN 1.50
                    WHEN value_risk_quadrant = 'Medium risk / high value' THEN 1.15
                    WHEN value_risk_quadrant = 'Low risk / high value' THEN 0.60
                    WHEN value_risk_quadrant = 'High risk / low value' THEN 0.25
                    ELSE 0.75
                  END AS lifecycle_model_priority_score,

            CASE
                WHEN value_risk_quadrant = 'High risk / high value'
                    AND future_churned_clv_proxy >= 1000000
                    THEN 'Highest priority lifecycle segment'
                WHEN value_risk_quadrant = 'Medium risk / high value'
                    AND future_churned_clv_proxy >= 1000000
                    THEN 'Secondary lifecycle segment'
                WHEN value_risk_quadrant = 'High risk / low value'
                    THEN 'Do not spend premium budget'
                WHEN value_risk_quadrant = 'Low risk / high value'
                    THEN 'Protect, but avoid discounting'
                ELSE 'Lifecycle monitoring segment'
            END AS lifecycle_model_recommendation
        FROM segment_base
        ORDER BY lifecycle_model_priority_score DESC
    """,

    "05_tableau_lifecycle_base": """
        SELECT
            lifecycle_stage,
            engagement_tier,
            revenue_tier,
            clv_value_tier,
            churn_risk_tier,
            value_risk_quadrant,
            lifecycle_strategy_readout,
            retention_budget_tier,
            COUNT(*) AS customers,
            AVG(churn_event) AS future_churn_rate,
            AVG(predicted_churn_probability) AS avg_predicted_churn_probability,
            AVG(lifecycle_duration_months) AS avg_lifecycle_duration_months,
            SUM(profit_adjusted_clv_proxy) AS profit_adjusted_clv_proxy,
            SUM(future_churned_clv_proxy) AS future_churned_clv_proxy,
            AVG(monthly_value_baseline) AS avg_monthly_value_baseline
        FROM customer_lifecycle_survival_base
        GROUP BY
            lifecycle_stage,
            engagement_tier,
            revenue_tier,
            clv_value_tier,
            churn_risk_tier,
            value_risk_quadrant,
            lifecycle_strategy_readout,
            retention_budget_tier
        HAVING COUNT(*) >= 250
        ORDER BY future_churned_clv_proxy DESC
    """,
}


def write_sql_file() -> None:
    sql_path = SQL_DIR / "06_survival_lifecycle_analysis.sql"
    with open(sql_path, "w") as f:
        f.write("-- customer_lifecycle_survival_base\n")
        f.write(LIFECYCLE_BASE_SQL.strip())
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


# KM-style curves provide timing intuition from observed lifecycle duration.
# They should be read as lifecycle-risk curves, not formal survival inference.
def kaplan_meier_curve(
    df: pd.DataFrame,
    group_cols=None,
    max_month: int = MAX_SURVIVAL_MONTH,
    min_group_size: int = 0,
) -> pd.DataFrame:
    if group_cols is None:
        groups = [("Overall", df)]
        group_cols = []
    else:
        groups = []
        for values, group in df.groupby(group_cols, dropna=False):
            if not isinstance(values, tuple):
                values = (values,)
            if len(group) >= min_group_size:
                groups.append((values, group))

    rows = []

    for group_values, group in groups:
        durations = group["lifecycle_duration_months"].astype(int)
        events = group["churn_event"].astype(int)

        max_t = int(min(max_month, max(durations.max(), 1)))
        survival = 1.0

        for month in range(1, max_t + 1):
            at_risk = int((durations >= month).sum())
            events_at_month = int(((durations == month) & (events == 1)).sum())
            censored_at_month = int(((durations == month) & (events == 0)).sum())

            hazard_rate = events_at_month / at_risk if at_risk > 0 else 0
            survival *= 1 - hazard_rate

            row = {
                "month_index": month,
                "customers_at_risk": at_risk,
                "churn_events": events_at_month,
                "censored_customers": censored_at_month,
                "hazard_rate": hazard_rate,
                "survival_probability": survival,
                "group_size": len(group),
            }

            if len(group_cols) == 0:
                row["group_label"] = "Overall"
            else:
                for col, val in zip(group_cols, group_values):
                    row[col] = val
                row["group_label"] = " / ".join(str(v) for v in group_values)

            rows.append(row)

    return pd.DataFrame(rows)


def survival_at_month(curve: pd.DataFrame, month: int, group_cols: list[str]) -> pd.DataFrame:
    records = []

    if not group_cols:
        sub = curve[curve["month_index"] <= month]
        survival = 1.0 if sub.empty else sub.sort_values("month_index").tail(1)["survival_probability"].iloc[0]
        return pd.DataFrame([{"group_label": "Overall", f"survival_at_month_{month}": survival}])

    for group_label, group in curve.groupby("group_label", dropna=False):
        sub = group[group["month_index"] <= month]
        survival = 1.0 if sub.empty else sub.sort_values("month_index").tail(1)["survival_probability"].iloc[0]

        row = {"group_label": group_label, f"survival_at_month_{month}": survival}
        for col in group_cols:
            row[col] = group[col].iloc[0]
        records.append(row)

    return pd.DataFrame(records)


def build_survival_timepoint_summary(
    lifecycle_df: pd.DataFrame,
    quadrant_curve: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = ["value_risk_quadrant"]

    base = (
        lifecycle_df.groupby(group_cols, as_index=False)
        .agg(
            customers=("msno", "count"),
            future_churn_rate=("churn_event", "mean"),
            avg_predicted_churn_probability=("predicted_churn_probability", "mean"),
            avg_lifecycle_duration_months=("lifecycle_duration_months", "mean"),
            profit_adjusted_clv_proxy=("profit_adjusted_clv_proxy", "sum"),
            future_churned_clv_proxy=("future_churned_clv_proxy", "sum"),
        )
    )

    s3 = survival_at_month(quadrant_curve, 3, group_cols)
    s6 = survival_at_month(quadrant_curve, 6, group_cols)
    s12 = survival_at_month(quadrant_curve, 12, group_cols)

    out = base.merge(s3[group_cols + ["survival_at_month_3"]], on=group_cols, how="left")
    out = out.merge(s6[group_cols + ["survival_at_month_6"]], on=group_cols, how="left")
    out = out.merge(s12[group_cols + ["survival_at_month_12"]], on=group_cols, how="left")

    return out.sort_values("future_churned_clv_proxy", ascending=False)


def validate_outputs(
    lifecycle_df: pd.DataFrame,
    survival_overall: pd.DataFrame,
    survival_quadrant: pd.DataFrame,
    results: dict[str, pd.DataFrame],
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
        "lifecycle_base_not_empty",
        len(lifecycle_df),
        len(lifecycle_df) > 0,
        "Lifecycle base should contain customer rows.",
    )

    duplicate_customers = lifecycle_df["msno"].duplicated().sum()

    add_check(
        "one_row_per_customer",
        int(duplicate_customers),
        duplicate_customers == 0,
        "Lifecycle survival base should have one row per customer.",
    )

    min_duration = lifecycle_df["lifecycle_duration_months"].min()

    add_check(
        "positive_lifecycle_duration",
        int(min_duration),
        min_duration >= 1,
        "Lifecycle duration should be at least one month.",
    )

    coverage_out_of_bounds = (
        (lifecycle_df["observed_month_coverage_rate"] < 0)
        | (lifecycle_df["observed_month_coverage_rate"] > 1)
    ).sum()

    add_check(
        "observed_month_coverage_rate_within_bounds",
        int(coverage_out_of_bounds),
        coverage_out_of_bounds == 0,
        "Observed month coverage should stay between 0 and 1.",
    )

    add_check(
        "overall_survival_curve_created",
        len(survival_overall),
        len(survival_overall) > 0,
        "Overall KM-style lifecycle-risk curve should exist.",
    )

    survival_non_increasing = (
        survival_overall["survival_probability"].diff().fillna(0) <= 1e-9
    ).all()

    add_check(
        "overall_survival_non_increasing",
        bool(survival_non_increasing),
        survival_non_increasing,
        "Survival probability should not increase over time.",
    )

    add_check(
        "value_risk_quadrant_curve_created",
        survival_quadrant["value_risk_quadrant"].nunique(),
        survival_quadrant["value_risk_quadrant"].nunique() >= 3,
        "Value/risk survival curves should cover multiple strategic groups.",
    )

    model_priority_rows = len(results["04_model_priority_lifecycle_segments"])

    add_check(
        "model_priority_segments_not_empty",
        model_priority_rows,
        model_priority_rows > 0,
        "Lifecycle model-priority segments should not be empty.",
    )

    report = pd.DataFrame(checks)
    report.to_csv(OUTPUT_DIR / "_survival_lifecycle_validation_report.csv", index=False)

    with open(OUTPUT_DIR / "_survival_lifecycle_validation_report.json", "w") as f:
        json.dump(checks, f, indent=2)

    print("\nValidation report:")
    print(report.to_string(index=False))

    failed = report[report["status"] == "FAIL"]
    if len(failed) > 0:
        raise ValueError("Survival/lifecycle validation failed. Review outputs before moving forward.")

    return report


def write_executive_summary(
    results: dict[str, pd.DataFrame],
    survival_overall: pd.DataFrame,
) -> None:
    portfolio = results["00_portfolio_lifecycle_summary"].iloc[0]
    lifecycle_stage = results["01_lifecycle_stage_summary"]
    hazard = results["02_monthly_hazard_by_observed_age"]
    quadrant = results["03_value_risk_quadrant_summary"]
    model_priority = results["04_model_priority_lifecycle_segments"]

    top_stage = lifecycle_stage.sort_values("future_churned_clv_proxy", ascending=False).iloc[0]
    peak_hazard = hazard.sort_values("monthly_hazard_rate", ascending=False).iloc[0]
    top_quadrant = quadrant.sort_values("future_churned_clv_proxy", ascending=False).iloc[0]
    top_model_segment = model_priority.sort_values("lifecycle_model_priority_score", ascending=False).iloc[0]

    overall_month_6 = survival_at_month(survival_overall, 6, [])[f"survival_at_month_6"].iloc[0]
    overall_month_12 = survival_at_month(survival_overall, 12, [])[f"survival_at_month_12"].iloc[0]

    summary = {
        "customers": int(portfolio["customers"]),
        "future_churn_rate": float(portfolio["future_churn_rate"]),
        "avg_lifecycle_duration_months": float(portfolio["avg_lifecycle_duration_months"]),
        "median_lifecycle_duration_months": float(portfolio["median_lifecycle_duration_months"]),
        "future_churned_clv_proxy": float(portfolio["future_churned_clv_proxy"]),
        "high_risk_high_value_customers": int(portfolio["high_risk_high_value_customers"]),
        "high_risk_high_value_future_churned_clv_proxy": float(
            portfolio["high_risk_high_value_future_churned_clv_proxy"]
        ),
        "overall_survival_month_6": float(overall_month_6),
        "overall_survival_month_12": float(overall_month_12),
        "top_lifecycle_stage": {
            "lifecycle_stage": str(top_stage["lifecycle_stage"]),
            "customers": int(top_stage["customers"]),
            "future_churn_rate": float(top_stage["future_churn_rate"]),
            "future_churned_clv_proxy": float(top_stage["future_churned_clv_proxy"]),
        },
        "peak_hazard_month": {
            "month_index": int(peak_hazard["month_index"]),
            "monthly_hazard_rate": float(peak_hazard["monthly_hazard_rate"]),
            "customers_at_risk": int(peak_hazard["customers_at_risk"]),
            "churn_events": int(peak_hazard["churn_events"]),
        },
        "top_value_risk_quadrant": {
            "value_risk_quadrant": str(top_quadrant["value_risk_quadrant"]),
            "strategy": str(top_quadrant["lifecycle_strategy_readout"]),
            "customers": int(top_quadrant["customers"]),
            "future_churn_rate": float(top_quadrant["future_churn_rate"]),
            "future_churned_clv_proxy": float(top_quadrant["future_churned_clv_proxy"]),
        },
        "top_model_priority_segment": {
            "lifecycle_stage": str(top_model_segment["lifecycle_stage"]),
            "engagement_tier": str(top_model_segment["engagement_tier"]),
            "revenue_tier": str(top_model_segment["revenue_tier"]),
            "clv_value_tier": str(top_model_segment["clv_value_tier"]),
            "churn_risk_tier": str(top_model_segment["churn_risk_tier"]),
            "customers": int(top_model_segment["customers"]),
            "future_churned_clv_proxy": float(top_model_segment["future_churned_clv_proxy"]),
            "recommendation": str(top_model_segment["lifecycle_model_recommendation"]),
        },
    }

    with open(OUTPUT_DIR / "_survival_lifecycle_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Survival and Lifecycle Analysis Summary",
        "",
        f"Customers analyzed: {summary['customers']:,}",
        f"Future churn rate in labeled snapshot: {summary['future_churn_rate']:.2%}",
        f"Average observed lifecycle duration: {summary['avg_lifecycle_duration_months']:.2f} months",
        f"Median observed lifecycle duration: {summary['median_lifecycle_duration_months']:.2f} months",
        "Interpretation: lifecycle duration is based on observed customer-month coverage, so survival outputs are lifecycle-risk proxies rather than exact churn event-time estimates.",
        f"Future churned CLV proxy: {summary['future_churned_clv_proxy']:,.0f}",
        "",
        "Lifecycle survival readout:",
        f"- Overall KM-style lifecycle survival proxy through month 6: {summary['overall_survival_month_6']:.2%}",
        f"- Overall KM-style lifecycle survival proxy through month 12: {summary['overall_survival_month_12']:.2%}",
        (
            f"- The highest observed hazard month is month {summary['peak_hazard_month']['month_index']}, "
            f"with a hazard rate of {summary['peak_hazard_month']['monthly_hazard_rate']:.2%}."
        ),
        "",
        "Value/risk concentration:",
        (
            f"- High-risk/high-value customers: {summary['high_risk_high_value_customers']:,}, "
            f"representing {summary['high_risk_high_value_future_churned_clv_proxy']:,.0f} future churned CLV proxy."
        ),
        (
            f"- The largest value-risk strategy bucket is **{summary['top_value_risk_quadrant']['value_risk_quadrant']} / "
            f"{summary['top_value_risk_quadrant']['strategy']}**, with "
            f"{summary['top_value_risk_quadrant']['customers']:,} customers and "
            f"{summary['top_value_risk_quadrant']['future_churned_clv_proxy']:,.0f} future churned CLV proxy."
        ),
        (
            f"- The lifecycle stage with the most future churned CLV proxy is "
            f"**{summary['top_lifecycle_stage']['lifecycle_stage']}**."
        ),
        "",
        "Modeling implication:",
        (
            f"The top lifecycle model-priority segment is **{summary['top_model_priority_segment']['lifecycle_stage']} / "
            f"{summary['top_model_priority_segment']['engagement_tier']} / "
            f"{summary['top_model_priority_segment']['revenue_tier']} / "
            f"{summary['top_model_priority_segment']['clv_value_tier']} / "
            f"{summary['top_model_priority_segment']['churn_risk_tier']}**. "
            "This is exactly the type of segment where the retention engine should avoid a simple rule-based offer "
            "and instead use save-worthiness scoring."
        ),
        "",
        "Business interpretation:",
        (
            "This layer adds timing context to the retention strategy. The model does not just rank customers by churn risk; "
            "it shows which lifecycle groups carry exposed value and where retention timing matters. This makes the next "
            "save-worthiness layer more defensible because it combines risk, value, and lifecycle stage."
        ),
        "",
        "Assumption note:",
        (
            "The KKBox label is a future churn outcome. These curves should be read as lifecycle-risk curves based on "
            "observed months and future churn labels, not as a perfect event-history survival study."
        ),
    ]

    summary_path = OUTPUT_DIR / "_executive_survival_lifecycle_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved {summary_path}")
    print(f"Saved {OUTPUT_DIR / '_survival_lifecycle_summary.json'}")


def main() -> None:
    print("\nRunning survival and lifecycle analysis...")

    if not CUSTOMER_MONTH_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CUSTOMER_MONTH_PATH}. Run src/01_build_customer_month_table.py first."
        )

    if not CHURN_SCORED_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CHURN_SCORED_PATH}. Run src/05_churn_model.py first."
        )

    con = duckdb.connect(database=":memory:")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE customer_month AS
        SELECT * FROM read_parquet('{CUSTOMER_MONTH_PATH}')
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE churn_scored_customers AS
        SELECT * FROM read_parquet('{CHURN_SCORED_PATH}')
        """
    )

    write_sql_file()

    con.execute(LIFECYCLE_BASE_SQL)

    con.execute(
        f"""
        COPY customer_lifecycle_survival_base
        TO '{OUTPUT_DIR / "customer_lifecycle_survival_features.parquet"}'
        (FORMAT PARQUET)
        """
    )

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM customer_lifecycle_survival_base
            ORDER BY predicted_churn_probability DESC, profit_adjusted_clv_proxy DESC
            LIMIT 100000
        )
        TO '{OUTPUT_DIR / "tableau_lifecycle_customer_sample.csv"}'
        (HEADER, DELIMITER ',')
        """
    )

    results = {}
    for name, query in SUMMARY_QUERIES.items():
        results[name] = export_query(con, name, query)

    lifecycle_df = con.execute("SELECT * FROM customer_lifecycle_survival_base").df()

    survival_overall = kaplan_meier_curve(lifecycle_df)
    survival_stage = kaplan_meier_curve(
        lifecycle_df,
        group_cols=["lifecycle_stage"],
        min_group_size=MIN_GROUP_SIZE,
    )
    survival_value = kaplan_meier_curve(
        lifecycle_df,
        group_cols=["clv_value_tier"],
        min_group_size=MIN_GROUP_SIZE,
    )
    survival_risk = kaplan_meier_curve(
        lifecycle_df,
        group_cols=["churn_risk_tier"],
        min_group_size=MIN_GROUP_SIZE,
    )
    survival_quadrant = kaplan_meier_curve(
        lifecycle_df,
        group_cols=["value_risk_quadrant"],
        min_group_size=MIN_GROUP_SIZE,
    )

    survival_timepoints = build_survival_timepoint_summary(lifecycle_df, survival_quadrant)

    survival_overall.to_csv(OUTPUT_DIR / "06_km_survival_overall.csv", index=False)
    survival_stage.to_csv(OUTPUT_DIR / "07_km_survival_by_lifecycle_stage.csv", index=False)
    survival_value.to_csv(OUTPUT_DIR / "08_km_survival_by_clv_value_tier.csv", index=False)
    survival_risk.to_csv(OUTPUT_DIR / "09_km_survival_by_churn_risk_tier.csv", index=False)
    survival_quadrant.to_csv(OUTPUT_DIR / "10_km_survival_by_value_risk_quadrant.csv", index=False)
    survival_timepoints.to_csv(OUTPUT_DIR / "11_survival_timepoint_summary.csv", index=False)

    print(f"Saved {OUTPUT_DIR / '06_km_survival_overall.csv'}")
    print(f"Saved {OUTPUT_DIR / '07_km_survival_by_lifecycle_stage.csv'}")
    print(f"Saved {OUTPUT_DIR / '08_km_survival_by_clv_value_tier.csv'}")
    print(f"Saved {OUTPUT_DIR / '09_km_survival_by_churn_risk_tier.csv'}")
    print(f"Saved {OUTPUT_DIR / '10_km_survival_by_value_risk_quadrant.csv'}")
    print(f"Saved {OUTPUT_DIR / '11_survival_timepoint_summary.csv'}")

    validate_outputs(
        lifecycle_df=lifecycle_df,
        survival_overall=survival_overall,
        survival_quadrant=survival_quadrant,
        results=results,
    )

    write_executive_summary(
        results=results,
        survival_overall=survival_overall,
    )

    print("\n06_survival_lifecycle_analysis.py complete.")


if __name__ == "__main__":
    main()
