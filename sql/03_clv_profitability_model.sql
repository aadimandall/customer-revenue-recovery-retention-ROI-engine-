-- customer_clv_scores
CREATE OR REPLACE TABLE customer_clv_scores AS
WITH base AS (
    SELECT
        ms.*,

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

        LEAST(
            24,
            GREATEST(
                1,
                3
                + (0.55 * engagement_decile)
                + CASE
                    WHEN auto_renew_signal = 1 THEN 4
                    ELSE 0
                  END
                + CASE
                    WHEN COALESCE(tenure_months, 0) >= 24 THEN 2
                    WHEN COALESCE(tenure_months, 0) >= 6 THEN 1
                    ELSE 0
                  END
                + CASE
                    WHEN cancellation_signal = 1 THEN -5
                    ELSE 0
                  END
                + CASE
                    WHEN COALESCE(major_activity_drop_flag, 0) = 1 THEN -2
                    ELSE 0
                  END
            )
        ) AS expected_active_months_proxy
    FROM ranked
),

clv_scored AS (
    SELECT
        *,

        0.65 AS gross_margin_rate,

        monthly_value_baseline * 0.65 AS monthly_margin_baseline,
        monthly_value_baseline * 12 AS annual_revenue_run_rate_proxy,
        monthly_value_baseline * 12 * 0.65 AS annual_margin_run_rate_proxy,

        monthly_value_baseline
            * 0.65
            * expected_active_months_proxy
            AS profit_adjusted_clv_proxy,

        CASE
            WHEN churn_next_period = 1 THEN
                monthly_value_baseline
                * 0.65
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

FROM clv_scored;

-- 00_portfolio_clv_summary
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
        FROM customer_clv_scores;

-- 01_clv_value_tier_summary
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
        ORDER BY total_profit_adjusted_clv_proxy DESC;

-- 02_clv_segment_summary
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
        ORDER BY future_churned_clv_proxy DESC;

-- 03_value_based_action_group_summary
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
        ORDER BY future_churned_clv_proxy DESC;

-- 04_high_value_recovery_targets
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
        LIMIT 100000;

-- 05_tableau_clv_segment_matrix
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
        ORDER BY future_churned_clv_proxy DESC;

