-- 01_monthly_revenue_leakage
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
        ORDER BY snapshot_month;

-- 02_cohort_retention_long
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
            a.months_since_first_observed;

-- 03_engagement_tier_churn_revenue
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
        ORDER BY future_churned_revenue_proxy DESC;

-- 04_revenue_tier_churn_revenue
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
        ORDER BY future_churned_revenue_proxy DESC;

-- 05_lifecycle_stage_churn_revenue
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
        ORDER BY future_churned_revenue_proxy DESC;

-- 06_registered_channel_churn_revenue
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
        ORDER BY future_churned_revenue_proxy DESC;

-- 07_retention_risk_segment_matrix
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
            churn_rate DESC;

-- 08_high_value_at_risk_segments
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
        LIMIT 30;

-- 09_tableau_cohort_heatmap
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
            months_since_first_observed;

