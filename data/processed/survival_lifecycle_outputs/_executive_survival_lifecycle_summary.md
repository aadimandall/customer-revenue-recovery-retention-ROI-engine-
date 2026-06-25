# Survival and Lifecycle Analysis Summary

Customers analyzed: 970,960
Future churn rate in labeled snapshot: 8.99%
Average observed lifecycle duration: 1.22 months
Median observed lifecycle duration: 1.00 months
Interpretation: lifecycle duration is based on observed customer-month coverage, so survival outputs are lifecycle-risk proxies rather than exact churn event-time estimates.
Future churned CLV proxy: 137,698,150

Lifecycle survival readout:
- Overall KM-style lifecycle survival proxy through month 6: 43.71%
- Overall KM-style lifecycle survival proxy through month 12: 35.18%
- The highest observed hazard month is month 3, with a hazard rate of 28.72%.

Value/risk concentration:
- High-risk/high-value customers: 179,814, representing 126,851,135 future churned CLV proxy.
- The largest value-risk strategy bucket is **High risk / high value / Immediate save-worthiness review**, with 44,304 customers and 120,377,400 future churned CLV proxy.
- The lifecycle stage with the most future churned CLV proxy is **Long-tenure**.

Modeling implication:
The top lifecycle model-priority segment is **Long-tenure / High engagement / High revenue / Elite value / Critical risk**. This is exactly the type of segment where the retention engine should avoid a simple rule-based offer and instead use save-worthiness scoring.

Business interpretation:
This layer adds timing context to the retention strategy. The model does not just rank customers by churn risk; it shows which lifecycle groups carry exposed value and where retention timing matters. This makes the next save-worthiness layer more defensible because it combines risk, value, and lifecycle stage.

Assumption note:
The KKBox label is a future churn outcome. These curves should be read as lifecycle-risk curves based on observed months and future churn labels, not as a perfect event-history survival study.