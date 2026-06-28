"""
01b_validate_customer_month_table.py

Validation layer for the customer-month table.

This script checks whether the customer-month and modeling snapshot outputs from
01_build_customer_month_table.py are structurally reliable before the project moves
into cohort retention, CLV, churn modeling, save-worthiness scoring, ROI simulation,
Tableau reporting, and Streamlit.

The goal is to catch data model issues early: duplicate customer-month rows, duplicate
snapshot rows, missing churn labels, invalid revenue values, and incomplete output files.

Outputs:
- data/processed/customer_month_validation_report.csv
- data/processed/customer_month_validation_report.json
- data/processed/customer_month_validation_report.txt

This file is intentionally separate from 01_build_customer_month_table.py so validation
can be rerun without rebuilding the full customer-month layer.
"""

from pathlib import Path
import json
import duckdb
import pandas as pd


PROCESSED_DIR = Path("data/processed")

CUSTOMER_MONTH_PATH = PROCESSED_DIR / "customer_month_table.parquet"
MODELING_SNAPSHOT_PATH = PROCESSED_DIR / "modeling_customer_snapshot.parquet"
TABLEAU_BASE_PATH = PROCESSED_DIR / "tableau_customer_retention_base.csv"

VALIDATION_CSV = PROCESSED_DIR / "customer_month_validation_report.csv"
VALIDATION_JSON = PROCESSED_DIR / "customer_month_validation_report.json"
VALIDATION_TXT = PROCESSED_DIR / "customer_month_validation_report.txt"


# Keep validation checks in a simple list-of-dicts format so the report can be
# written to CSV, JSON, and text without separate formatting logic.
def add_check(checks, check_name, value, pass_condition, notes):
    status = "PASS" if pass_condition else "FAIL"
    checks.append(
        {
            "check_name": check_name,
            "status": status,
            "value": value,
            "notes": notes,
        }
    )


def main() -> None:
    print("\nValidating customer-month outputs...")

    # Fail fast if the build step did not create the expected downstream files.
    missing_files = [
        str(path)
        for path in [CUSTOMER_MONTH_PATH, MODELING_SNAPSHOT_PATH, TABLEAU_BASE_PATH]
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(f"Missing required output files: {missing_files}")

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

    checks = []

    # Basic grain checks: customer_month can have many rows per customer,
    # while modeling_snapshot should have exactly one row per labeled customer.
    customer_month_rows = con.execute("SELECT COUNT(*) FROM customer_month").fetchone()[0]
    customer_month_users = con.execute("SELECT COUNT(DISTINCT msno) FROM customer_month").fetchone()[0]
    snapshot_rows = con.execute("SELECT COUNT(*) FROM modeling_snapshot").fetchone()[0]
    snapshot_users = con.execute("SELECT COUNT(DISTINCT msno) FROM modeling_snapshot").fetchone()[0]

    add_check(
        checks,
        "customer_month_table_not_empty",
        customer_month_rows,
        customer_month_rows > 0,
        "Customer-month table should have rows.",
    )

    add_check(
        checks,
        "modeling_snapshot_not_empty",
        snapshot_rows,
        snapshot_rows > 0,
        "Modeling snapshot should have rows.",
    )

    # Duplicate customer-month rows would break cohort analysis and rolling features.
    duplicate_customer_months = con.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT msno, snapshot_month, COUNT(*) AS row_count
            FROM customer_month
            GROUP BY msno, snapshot_month
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    add_check(
        checks,
        "no_duplicate_customer_month_rows",
        duplicate_customer_months,
        duplicate_customer_months == 0,
        "There should be only one row per customer per observed month.",
    )

    # The snapshot is the decision table used by CLV, churn, and ROI scripts,
    # so duplicate customers here would duplicate downstream recommendations.
    duplicate_snapshot_users = con.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT msno, COUNT(*) AS row_count
            FROM modeling_snapshot
            GROUP BY msno
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    add_check(
        checks,
        "one_snapshot_row_per_customer",
        duplicate_snapshot_users,
        duplicate_snapshot_users == 0,
        "The modeling snapshot should have one row per labeled customer.",
    )

    add_check(
        checks,
        "snapshot_rows_equal_unique_users",
        f"rows={snapshot_rows}, unique_users={snapshot_users}",
        snapshot_rows == snapshot_users,
        "Snapshot row count should match distinct customer count.",
    )

    # Every snapshot row needs a churn label because the next modeling layer is supervised.
    missing_target_rows = con.execute(
        """
        SELECT COUNT(*)
        FROM modeling_snapshot
        WHERE churn_next_period IS NULL
        """
    ).fetchone()[0]

    add_check(
        checks,
        "no_missing_churn_target",
        missing_target_rows,
        missing_target_rows == 0,
        "Every modeling row should have a churn label.",
    )

    churn_rate = con.execute(
        """
        SELECT AVG(churn_next_period)
        FROM modeling_snapshot
        """
    ).fetchone()[0]

    add_check(
        checks,
        "churn_rate_reasonable",
        round(float(churn_rate), 6),
        0 < churn_rate < 0.5,
        "Churn rate should be non-zero and below 50% for this KKBox label window.",
    )

    negative_revenue_rows = con.execute(
        """
        SELECT COUNT(*)
        FROM customer_month
        WHERE monthly_revenue < 0
           OR monthly_list_price < 0
        """
    ).fetchone()[0]

    add_check(
        checks,
        "no_negative_revenue_values",
        negative_revenue_rows,
        negative_revenue_rows == 0,
        "Monthly revenue and list price should not be negative.",
    )

    invalid_tenure_rows = con.execute(
        """
        SELECT COUNT(*)
        FROM customer_month
        WHERE tenure_months < -1
        """
    ).fetchone()[0]

    add_check(
        checks,
        "tenure_values_reasonable",
        invalid_tenure_rows,
        invalid_tenure_rows == 0,
        "Tenure should not be strongly negative. Small edge cases may come from date alignment.",
    )

    # This is informational, not a failure. Some labeled customers have no observed
    # transaction/activity month after filtering, but they are still kept in the snapshot.
    users_without_observed_month = con.execute(
        """
        SELECT COUNT(*)
        FROM modeling_snapshot
        WHERE snapshot_month = 'no_observed_month'
        """
    ).fetchone()[0]

    add_check(
        checks,
        "users_without_observed_month_tracked",
        users_without_observed_month,
        users_without_observed_month >= 0,
        "This is an informational check showing labeled users without observed transaction/activity rows.",
    )

    month_range = con.execute(
        """
        SELECT MIN(snapshot_month), MAX(snapshot_month)
        FROM customer_month
        """
    ).fetchone()

    add_check(
        checks,
        "customer_month_date_range_exists",
        f"{month_range[0]} to {month_range[1]}",
        month_range[0] is not None and month_range[1] is not None,
        "Customer-month table should have a valid observed month range.",
    )

    tableau_file_size = TABLEAU_BASE_PATH.stat().st_size

    add_check(
        checks,
        "tableau_base_file_exists",
        tableau_file_size,
        tableau_file_size > 0,
        "CSV export for Tableau should exist and be non-empty.",
    )

    # Persist validation results so the project has an audit trail before modeling starts.
    report = pd.DataFrame(checks)
    report.to_csv(VALIDATION_CSV, index=False)

    with open(VALIDATION_JSON, "w") as f:
        json.dump(checks, f, indent=2)

    with open(VALIDATION_TXT, "w") as f:
        for check in checks:
            f.write(
                f"{check['status']} | {check['check_name']} | "
                f"value={check['value']} | {check['notes']}\n"
            )

    print("\nValidation report:")
    print(report.to_string(index=False))

    failed = report[report["status"] == "FAIL"]

    print("\nSaved outputs:")
    print(VALIDATION_CSV)
    print(VALIDATION_JSON)
    print(VALIDATION_TXT)

    if len(failed) > 0:
        print("\nValidation finished with failures. Review before moving forward.")
        raise SystemExit(1)

    print("\nAll structural validation checks passed.")
    print("01b_validate_customer_month_table.py complete.")


if __name__ == "__main__":
    main()
