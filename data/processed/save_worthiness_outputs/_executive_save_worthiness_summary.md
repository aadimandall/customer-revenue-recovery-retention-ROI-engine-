# Save-Worthiness Scoring Summary

Customers scored: 970,960
Economically positive paid-offer candidates: 71,437
Paid-offer candidate rate: 7.36%
Expected saved CLV proxy from paid-offer pool: 16,620,209
Estimated intervention cost proxy for paid-offer pool: 791,284
Expected net save value proxy: 15,828,925
Paid-offer ROI proxy: 20.00x

Key readout:
- The highest expected-value action is **Immediate premium save offer**, with 37,303 customers and 15,527,916 expected net save value proxy.
- The strongest priority tier is **Tier 2 - Premium save priority**, with 2,857 customers and 4,399,204 expected net save value proxy.
- The top save-worthiness segment is **Long-tenure / High engagement / High revenue / Elite value / Critical risk / Premium save budget**, with 2,068 customers.
- The largest suppression reason is **Monitor only before ROI testing**, covering 722,108 customers.

Business interpretation:
This layer turns churn prediction into a retention decision engine. It avoids the common mistake of targeting customers only because they have high churn probability. The score prioritizes customers where churn risk, customer value, expected response, and intervention cost create positive expected economics.

Next step:
The next script should run retention ROI simulation. It should stress-test response rates, discount costs, campaign budgets, and targeting thresholds before recommending a campaign strategy.

Assumption note:
Response rates and intervention costs are planning assumptions. They are intentionally separated into this layer so the ROI simulator can test conservative, base, and aggressive scenarios later.