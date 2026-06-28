-- customer_lifecycle_survival_base
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

FROM base;

-- 00_portfolio_lifecycle_summary
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
        FROM customer_lifecycle_survival_base;

-- 01_lifecycle_stage_summary
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
        ORDER BY future_churned_clv_proxy DESC;

-- 02_monthly_hazard_by_observed_age
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
        ORDER BY m.month_index;

-- 03_value_risk_quadrant_summary
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
        ORDER BY future_churned_clv_proxy DESC;

-- 04_model_priority_lifecycle_segments
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
        ORDER BY lifecycle_model_priority_score DESC;

-- 05_tableau_lifecycle_base
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
        ORDER BY future_churned_clv_proxy DESC;

