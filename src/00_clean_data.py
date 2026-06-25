"""
00_clean_data.py

Initial data cleaning layer for the Customer Revenue Recovery & Retention ROI Engine.

This script loads the KKBox churn files, filters the raw tables to the labeled
training population, standardizes dates and fields, and creates clean interim
tables for the customer-month model.

Raw KKBox files are intentionally not committed to GitHub.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_yyyymmdd(series: pd.Series) -> pd.Series:
    """Convert KKBox YYYYMMDD integer date fields into pandas datetime."""
    return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)
    print(f"Saved {path} | rows={len(df):,} cols={df.shape[1]:,}")


def clean_train() -> pd.DataFrame:
    print("\nLoading train_v2.csv...")
    train = pd.read_csv(RAW_DIR / "train_v2.csv")

    train["msno"] = train["msno"].astype(str)
    train["is_churn"] = train["is_churn"].astype(int)

    save_parquet(train, INTERIM_DIR / "train_clean.parquet")
    return train


def clean_members(train_users: set[str]) -> pd.DataFrame:
    print("\nCleaning members_v3.csv in chunks...")

    chunks = []
    for i, chunk in enumerate(pd.read_csv(RAW_DIR / "members_v3.csv", chunksize=1_000_000), start=1):
        chunk["msno"] = chunk["msno"].astype(str)
        chunk = chunk[chunk["msno"].isin(train_users)].copy()

        if chunk.empty:
            continue

        chunk["registration_date"] = parse_yyyymmdd(chunk["registration_init_time"])
        chunk["gender"] = chunk["gender"].fillna("unknown").astype(str)

        # KKBox has many missing or placeholder ages. Keep realistic ages only.
        chunk["age"] = pd.to_numeric(chunk["bd"], errors="coerce")
        chunk.loc[(chunk["age"] <= 0) | (chunk["age"] > 100), "age"] = np.nan

        chunk["city"] = chunk["city"].fillna(0).astype(int)
        chunk["registered_via"] = chunk["registered_via"].fillna(0).astype(int)

        chunk = chunk[
            [
                "msno",
                "city",
                "age",
                "gender",
                "registered_via",
                "registration_date",
            ]
        ]

        chunks.append(chunk)
        print(f"  processed member chunk {i} | kept rows={len(chunk):,}")

    members = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    members = members.drop_duplicates(subset=["msno"], keep="last")

    save_parquet(members, INTERIM_DIR / "members_clean.parquet")
    return members


def clean_transactions(train_users: set[str]) -> pd.DataFrame:
    print("\nCleaning transactions_v2.csv...")

    tx = pd.read_csv(RAW_DIR / "transactions_v2.csv")
    tx["msno"] = tx["msno"].astype(str)
    tx = tx[tx["msno"].isin(train_users)].copy()

    tx["transaction_date_dt"] = parse_yyyymmdd(tx["transaction_date"])
    tx["membership_expire_date_dt"] = parse_yyyymmdd(tx["membership_expire_date"])

    tx["transaction_month"] = tx["transaction_date_dt"].dt.to_period("M").astype(str)
    tx["expire_month"] = tx["membership_expire_date_dt"].dt.to_period("M").astype(str)

    numeric_cols = [
        "payment_method_id",
        "payment_plan_days",
        "plan_list_price",
        "actual_amount_paid",
        "is_auto_renew",
        "is_cancel",
    ]

    for col in numeric_cols:
        tx[col] = pd.to_numeric(tx[col], errors="coerce").fillna(0)

    tx["discount_amount"] = tx["plan_list_price"] - tx["actual_amount_paid"]
    tx["is_discounted"] = (tx["discount_amount"] > 0).astype(int)
    tx["paid_per_day"] = np.where(
        tx["payment_plan_days"] > 0,
        tx["actual_amount_paid"] / tx["payment_plan_days"],
        0,
    )

    tx = tx.sort_values(["msno", "transaction_date_dt", "membership_expire_date_dt"])

    save_parquet(tx, INTERIM_DIR / "transactions_clean.parquet")
    return tx


def clean_user_logs_monthly(train_users: set[str]) -> pd.DataFrame:
    print("\nAggregating user_logs_v2.csv to user-month level...")

    log_cols = [
        "msno",
        "date",
        "num_25",
        "num_50",
        "num_75",
        "num_985",
        "num_100",
        "num_unq",
        "total_secs",
    ]

    monthly_parts = []

    for i, chunk in enumerate(
        pd.read_csv(RAW_DIR / "user_logs_v2.csv", usecols=log_cols, chunksize=2_000_000),
        start=1,
    ):
        chunk["msno"] = chunk["msno"].astype(str)
        chunk = chunk[chunk["msno"].isin(train_users)].copy()

        if chunk.empty:
            print(f"  processed log chunk {i} | kept rows=0")
            continue

        chunk["date_dt"] = parse_yyyymmdd(chunk["date"])
        chunk["activity_month"] = chunk["date_dt"].dt.to_period("M").astype(str)

        play_cols = ["num_25", "num_50", "num_75", "num_985", "num_100"]
        for col in play_cols + ["num_unq", "total_secs"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)

        chunk["total_plays"] = chunk[play_cols].sum(axis=1)

        monthly = (
            chunk.groupby(["msno", "activity_month"], as_index=False)
            .agg(
                active_days=("date", "nunique"),
                num_25=("num_25", "sum"),
                num_50=("num_50", "sum"),
                num_75=("num_75", "sum"),
                num_985=("num_985", "sum"),
                num_100=("num_100", "sum"),
                num_unq=("num_unq", "sum"),
                total_secs=("total_secs", "sum"),
                total_plays=("total_plays", "sum"),
            )
        )

        monthly_parts.append(monthly)
        print(f"  processed log chunk {i} | kept rows={len(chunk):,} | monthly rows={len(monthly):,}")

    if not monthly_parts:
        logs_monthly = pd.DataFrame()
    else:
        logs_monthly = pd.concat(monthly_parts, ignore_index=True)

        logs_monthly = (
            logs_monthly.groupby(["msno", "activity_month"], as_index=False)
            .agg(
                active_days=("active_days", "sum"),
                num_25=("num_25", "sum"),
                num_50=("num_50", "sum"),
                num_75=("num_75", "sum"),
                num_985=("num_985", "sum"),
                num_100=("num_100", "sum"),
                num_unq=("num_unq", "sum"),
                total_secs=("total_secs", "sum"),
                total_plays=("total_plays", "sum"),
            )
        )

    logs_monthly["completion_rate_proxy"] = np.where(
        logs_monthly["total_plays"] > 0,
        logs_monthly["num_100"] / logs_monthly["total_plays"],
        0,
    )

    logs_monthly["engagement_score"] = (
        np.log1p(logs_monthly["total_secs"])
        + 0.25 * np.log1p(logs_monthly["num_unq"])
        + 0.50 * np.log1p(logs_monthly["active_days"])
    )

    save_parquet(logs_monthly, INTERIM_DIR / "user_logs_monthly.parquet")
    return logs_monthly


def write_data_quality_summary(train, members, transactions, logs_monthly) -> None:
    print("\nWriting data quality summary...")

    summary = {
        "train_rows": int(len(train)),
        "train_unique_users": int(train["msno"].nunique()),
        "churn_rate": float(train["is_churn"].mean()),
        "members_rows_filtered_to_train_users": int(len(members)),
        "transactions_rows_filtered_to_train_users": int(len(transactions)),
        "transaction_unique_users": int(transactions["msno"].nunique()),
        "user_log_monthly_rows": int(len(logs_monthly)),
        "user_log_unique_users": int(logs_monthly["msno"].nunique()),
        "raw_files_used": [
            "members_v3.csv",
            "train_v2.csv",
            "transactions_v2.csv",
            "user_logs_v2.csv",
        ],
    }

    with open(PROCESSED_DIR / "data_quality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(PROCESSED_DIR / "data_quality_summary.txt", "w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print(json.dumps(summary, indent=2))


def main() -> None:
    train = clean_train()
    train_users = set(train["msno"])

    members = clean_members(train_users)
    transactions = clean_transactions(train_users)
    logs_monthly = clean_user_logs_monthly(train_users)

    write_data_quality_summary(train, members, transactions, logs_monthly)

    print("\n00_clean_data.py complete.")


if __name__ == "__main__":
    main()
