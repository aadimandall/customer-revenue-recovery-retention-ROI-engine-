# CLV and Profitability Summary

Customers scored: 970,960
Future churn rate in labeled snapshot: 8.99%
Gross margin assumption: 65%

Portfolio value proxy:
- Annual revenue run-rate proxy: 1,694,883,728
- Annual margin run-rate proxy: 1,101,674,423
- Total profit-adjusted CLV proxy: 974,749,752
- Future churned CLV proxy: 137,394,802

Key readout:
- The largest value tier by total CLV is **Elite value**, with 194,192 customers and 17.66% future churn.
- The action group with the most future churned CLV proxy is **Protect high-value customer**, representing 159,820 customers and 106,109,424 future churned CLV proxy.

Business interpretation:
This layer separates customer value from customer churn risk. That matters because the retention engine should not simply target the highest-risk customers. It should prioritize customers where the expected saved margin justifies the intervention cost.

Assumption note:
This is a pre-model CLV proxy based on observed revenue, margin assumption, tenure, engagement, auto-renew behavior, cancellation signals, and activity decline. The next stage will build a churn-risk model and combine predicted churn probability with this value layer.