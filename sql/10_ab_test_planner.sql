-- 10_ab_test_planner.sql
-- SQL reference for A/B test planning outputs.

-- Treatment/control balance
SELECT
    experiment_group,
    customers,
    avg_predicted_churn_probability,
    actual_future_churn_rate_retrospective,
    gross_value_at_risk_proxy,
    planned_campaign_cost_proxy
FROM read_csv_auto('data/processed/ab_test_outputs/03_treatment_control_plan.csv')
ORDER BY experiment_group;

-- Experiment design summary
SELECT
    experiment_name,
    candidate_customers,
    treatment_customers,
    control_customers,
    control_churn_rate_planning,
    treatment_churn_rate_planning,
    expected_absolute_churn_reduction,
    treatment_expected_net_save_value_proxy,
    treatment_roi_proxy,
    approx_power_for_base_effect
FROM read_csv_auto('data/processed/ab_test_outputs/01_experiment_design_summary.csv');

-- Power scenarios
SELECT
    scenario_label,
    target_power,
    required_sample_size_per_group,
    planned_sample_size_per_group,
    approx_power_with_planned_sample,
    power_readout
FROM read_csv_auto('data/processed/ab_test_outputs/02_power_sample_size_scenarios.csv')
ORDER BY scenario_label, target_power;