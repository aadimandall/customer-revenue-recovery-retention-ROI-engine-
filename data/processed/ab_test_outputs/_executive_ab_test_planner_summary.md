# A/B Test Planner Summary

Recommended experiment: Save-worthy retention offer A/B test

Candidate customers: 62,836
Treatment customers: 31,430
Control customers: 31,406
Recommended rollout budget cap proxy: 750,000
Efficient budget value capture vs max tested net value: 99.99%

Expected test impact:
- Planning control churn rate: 64.81%
- Planning treatment churn rate: 57.40%
- Expected absolute churn reduction: 7.41%
- Expected relative churn reduction: 11.43%
- Expected customers saved in treatment arm: 2,332
- Treatment expected net save value proxy: 7,933,466
- Treatment campaign cost proxy: 375,121
- Treatment ROI proxy: 21.15x

Power readout:
- Approximate power for base expected effect: 100.00%
- Required sample per group for 80% power under base response: 677
- Planned sample per group: 31,418

Break-even readout:
- Break-even response rate: 0.61%

Decision rules:
- Scale only if treatment reduces churn, creates positive incremental net value, and passes customer-experience guardrails.
- Do not scale from churn reduction alone.
- Kill or redesign the campaign if economics fail, even if the model ranking looks strong.

Human launch-readiness readout:
- The test uses deterministic stratified randomization, not a naive global split.
- Balance diagnostics check whether treatment and control are comparable before launch.
- Minimum detectable effect scenarios show what size of churn reduction the test can realistically detect.
- The risk register documents response uncertainty, cannibalization, customer experience, implementation leakage, and timing bias.

Business interpretation:
This final layer converts the retention engine into a testable operating plan. The project now moves from model output to accountable experimentation: who gets the offer, who is held out, what lift is required, how much value is at stake, what risks could invalidate the test, and what rules determine rollout.