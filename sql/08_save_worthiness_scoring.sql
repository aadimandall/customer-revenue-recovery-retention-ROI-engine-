-- 08_save_worthiness_scoring.sql
-- SQL reference for the save-worthiness decision engine.

-- Core scoring idea:
-- expected_saved_clv_proxy = predicted_churn_probability * profit_adjusted_clv_proxy * expected_response_rate
-- net_save_value_proxy = expected_saved_clv_proxy - intervention_cost_proxy

SELECT
    msno,
    predicted_churn_probability,
    profit_adjusted_clv_proxy,
    expected_intervention_response_rate,
    intervention_cost_proxy,
    expected_saved_clv_proxy,
    net_save_value_proxy,
    save_worthiness_score,
    save_priority_tier,
    recommended_retention_action
FROM read_parquet('data/processed/save_worthiness_outputs/customer_save_worthiness_scores.parquet')
ORDER BY net_save_value_proxy DESC;

-- Recommended action summary
SELECT
    recommended_retention_action,
    COUNT(*) AS customers,
    SUM(expected_saved_clv_proxy) AS expected_saved_clv_proxy,
    SUM(intervention_cost_proxy) AS intervention_cost_proxy,
    SUM(net_save_value_proxy) AS net_save_value_proxy
FROM read_parquet('data/processed/save_worthiness_outputs/customer_save_worthiness_scores.parquet')
GROUP BY recommended_retention_action
ORDER BY net_save_value_proxy DESC;