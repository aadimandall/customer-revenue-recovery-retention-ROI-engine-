# Model Validation and Business Impact Summary

Best model: hist_gradient_boosting
Business validation grade: Excellent business ranking model
ROC-AUC: 0.9808
PR-AUC: 0.9151
PR-AUC multiple vs base churn rate: 10.17x
Brier score: 0.0198

Business lift validation:
- Top risk decile lift: 8.86x
- Top risk decile captures 88.58% of churners.
- Top risk decile captures 92.20% of future churned CLV proxy.

Recommended planning threshold:
- Use the top 5% risk population as the planning reference before save-worthiness scoring.
- Targeted customers at that point: 12,137
- Churn capture rate: 54.38%
- Future churned CLV capture rate: 83.34%

Calibration readout:
- Average absolute calibration gap: 0.07%
- Maximum absolute calibration gap: 0.25%
- Calibration is strong enough for planning.

Highest business-impact validation group:
- Critical risk / Elite value / Premium save budget contains 37,303 customers and 116,026,254 future churned CLV proxy.

Highest-priority segment for the next layer:
- Long-tenure / High engagement / High revenue / Elite value / Critical risk / Premium save budget
- Recommendation: Highest priority for save-worthiness layer

Cancellation-signal sensitivity:
- The model still ranks churn risk without direct cancellation signals. Cancellation behavior helps, but broader engagement and transaction patterns also matter.

Feature-importance governance:
- Top permutation-importance driver: trailing_3mo_revenue (Revenue and payment behavior).
- Direct cancellation-signal importance share: 30.20%
- Feature importance is not dominated by cancellation signals. The model appears to learn broader customer behavior patterns.

Governance readout:
- Governance status: PASS
- The model is validated as a churn-risk ranking engine. It is not the final retention action list. The next script should combine predicted churn probability with CLV, margin, intervention cost, expected response, and lifecycle context.