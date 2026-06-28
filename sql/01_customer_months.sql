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
    -- Use the union of transaction months and activity months so customers are
    -- kept when they have listening activity but no payment, or payment but no logs.
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

        -- The churn label is carried through for supervised modeling later.
        -- It is not used to create the behavior features in this table.
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
        -- Prior-month and trailing-window features capture behavior change,
        -- which is usually more useful for retention than a single raw month.
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

CREATE OR REPLACE TABLE modeling_customer_snapshot AS
WITH latest_customer_month AS (
    SELECT
        *,
        -- Keep the most recent observed customer-month as the current state.
        ROW_NUMBER() OVER (
            PARTITION BY msno
            ORDER BY snapshot_month_date DESC
        ) AS rn
    FROM customer_month_table
),

latest_transaction AS (
    SELECT
        msno,
        -- Latest transaction fields capture the most recent payment and cancel signals.
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

    -- Some labeled users may not have transaction/activity coverage after filtering.
    -- Keep them in the modeling snapshot so the row count still matches train_v2.
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

    -- These simple flags become interpretable churn and save-worthiness signals.
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
