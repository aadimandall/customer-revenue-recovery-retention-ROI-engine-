# Revenue Leakage and Value Concentration Summary

Customers analyzed: 970,960
Future churn rate in labeled snapshot: 8.99%
Annual revenue run-rate proxy: 1,694,883,728
Annual margin run-rate proxy: 1,101,674,423
Profit-adjusted CLV proxy: 974,361,046
Future churned CLV proxy: 137,698,150
CLV leakage rate: 14.13%

Key readout:
- The largest value-loss action group is **Protect high-value customer**, with 159,813 customers and 106,082,655 future churned CLV proxy.
- The top leakage driver segment is **Long-tenure / High engagement / High revenue / Elite value**, with 54,901 customers and 62,930,293 future churned CLV proxy.
- The top 5% of customers represent 16.81% of CLV and 78.09% of future churned CLV proxy.
- The top 10% of customers represent 23.89% of CLV and 79.63% of future churned CLV proxy.
- At a 10% base recovery assumption, recoverable portfolio CLV opportunity is 13,769,815.
- The strongest budget opportunity is **Premium save budget**, with estimated net recovery opportunity of 11,586,261.

Business interpretation:
The leakage is not evenly distributed across the customer base. High-value customers create a disproportionate share of exposed value, which means a generic churn campaign would waste money. The next model should not optimize for churn probability alone; it should identify where churn risk, customer value, and intervention economics overlap.

Modeling implication:
The top model-priority segment is **Long-tenure / High engagement / High revenue / Elite value**. The churn model should help separate customers who need intervention from customers who are high value but likely to stay without a paid offer.

Assumption note:
This script uses the future churn label for retrospective leakage diagnosis and opportunity sizing. It does not create the final targeting list. Final targeting is created later using predicted churn probability.