-- 07_model_validation_business_impact.sql
-- SQL reference queries for model validation outputs.
-- The Python script creates the final CSVs, but these queries document the business logic.

-- Top risk decile lift and value capture
SELECT
    risk_decile,
    customers,
    observed_churn_rate,
    lift_vs_portfolio,
    churn_capture_share,
    cumulative_churn_capture_share,
    future_churned_clv_share,
    cumulative_future_churned_clv_share
FROM read_csv_auto('data/processed/churn_model_outputs/01_risk_decile_business_summary.csv')
ORDER BY risk_decile;

-- Operating threshold policy simulation
SELECT
    target_population_pct,
    targeted_customers,
    score_threshold,
    observed_churn_rate_targeted,
    lift_vs_portfolio,
    churn_capture_rate,
    future_churned_clv_capture_rate
FROM read_csv_auto('data/processed/churn_model_outputs/02_targeting_threshold_simulation.csv')
ORDER BY target_population_pct;

-- Calibration diagnostic
SELECT
    calibration_bin,
    customers,
    avg_predicted_churn_probability,
    observed_churn_rate,
    calibration_gap
FROM read_csv_auto('data/processed/churn_model_outputs/03_calibration_by_score_bin.csv')
ORDER BY calibration_bin;

-- Risk tier and value concentration after scoring
SELECT
    churn_risk_tier,
    clv_value_tier,
    retention_budget_tier,
    customers,
    avg_predicted_churn_probability,
    actual_future_churn_rate,
    profit_adjusted_clv_proxy,
    future_churned_clv_proxy
FROM read_csv_auto('data/processed/model_validation_outputs/05_risk_tier_value_impact.csv')
ORDER BY future_churned_clv_proxy DESC;