# Churn Model Summary

Best model: hist_gradient_boosting
Model note: Main nonlinear churn-risk model.
Uses direct cancellation signals: True
ROC-AUC: 0.9808
PR-AUC: 0.9151
Brier score: 0.0198
Log loss: 0.0764
Test churn rate: 8.99%

Business lift:
- Top risk decile observed churn rate: 79.67%
- Top risk decile lift: 8.86x
- Top risk decile captures 88.58% of churners.
- Top risk decile captures 92.20% of future churned CLV proxy.

Planning operating point:
- Targeting the top 20% risk segment captures 95.78% of churners.
- Targeting the top 20% risk segment captures 96.94% of future churned CLV proxy.

Interpretation:
This model ranks customers by churn risk. It is not the final retention targeting list. The next layers combine predicted churn probability with profit-adjusted CLV, save cost, expected response, and campaign ROI.

Leakage control:
Future-churned value fields and final value/action fields were excluded from model features. They are used only after scoring for retrospective business lift analysis.