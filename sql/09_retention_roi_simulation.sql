-- 09_retention_roi_simulation.sql
-- SQL reference for retention ROI simulation outputs.

-- Strategy/scenario matrix
SELECT
    strategy_name,
    scenario_name,
    targeted_customers,
    expected_customers_saved,
    expected_saved_clv_proxy,
    campaign_cost_proxy,
    net_save_value_proxy,
    roi_proxy,
    budget_status,
    recommended_decision
FROM read_csv_auto('data/processed/retention_roi_outputs/01_strategy_scenario_matrix.csv')
ORDER BY scenario_name, net_save_value_proxy DESC;

-- Budget frontier
SELECT
    budget_cap,
    targeted_customers,
    campaign_cost_proxy,
    expected_saved_clv_proxy,
    net_save_value_proxy,
    roi_proxy
FROM read_csv_auto('data/processed/retention_roi_outputs/03_budget_cap_frontier.csv')
ORDER BY budget_cap;

-- Break-even response analysis
SELECT
    strategy_name,
    scenario_name,
    adjusted_expected_response_rate,
    break_even_response_rate,
    response_headroom,
    break_even_readout
FROM read_csv_auto('data/processed/retention_roi_outputs/07_break_even_summary.csv')
ORDER BY scenario_name, net_save_value_proxy DESC;