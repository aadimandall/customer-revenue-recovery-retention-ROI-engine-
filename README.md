# Customer Revenue Recovery & Retention ROI Engine

## Interactive Tableau Dashboard

View the executive Tableau dashboard here: [Customer Revenue Recovery & Retention ROI Dashboard](https://public.tableau.com/views/CustomerRevenueRecoveryRetentionROIDashboard/EXECUTIVEDECISIONBOARD?:language=en-GB&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

This Tableau Public dashboard presents the project as an executive retention decision board, showing customer value leakage, priority retention segments, save-worthiness triage, ROI budget frontier, A/B test readiness, and model governance.

Most churn projects start with the same question.

Who is likely to leave?

I started there too, but I did not want the project to end there. A churn score is useful, but it is not a retention strategy. It does not tell a company whether a customer deserves a paid offer, whether the offer clears cost, or whether the campaign would create value that would not have happened anyway.

That gap became the center of the project.

I built this as a decision engine because the real retention problem is not only predictive. It is economic. A customer can have high churn risk and still be the wrong person to target. Another customer can be valuable but not need a discount. Some customers may accept an offer even though they would have stayed. Others are easy to predict only because they are already showing cancellation behavior. Those cases are messy, but they are also what make the problem realistic.

The engine moves through the full decision chain: revenue leakage, customer value, churn risk, lifecycle context, save-worthiness, ROI simulation, and A/B test design.

The question I kept returning to was:

**Who should we save, what should we spend, and how do we prove it worked?** 

## Design principle

I treated each layer as a decision filter.

Churn modeling identifies risk. CLV adds customer value. Lifecycle context shows where a customer sits in the relationship. Save-worthiness scoring tests whether intervention clears cost. ROI simulation checks whether strategy still works under budget and response assumptions. A/B test planning turns the recommendation into something leadership can prove before scaling.

That structure was intentional. I did not want a model score to trigger action by itself. In real retention work, budget is limited, and high-risk customers are not always the best customers to target.

The core question became:

Where does the next dollar of retention spend have the best chance of protecting value?



## How the engine works

This project is structured as a retention decision engine, not a standalone churn model.

The goal is to connect customer risk, customer value, campaign economics, and test design so the final output is not just a prediction, but a strategy that can be tested before rollout.

The first step was building a customer table that could be trusted. Before any modeling, I created a one-row-per-customer foundation and checked for duplicate records, missing CLV values, negative value metrics, and incomplete strategy assignments. Every customer needed a value tier, action group, and budget tier so the analysis could support business decisions instead of only producing model scores.

After the data foundation was stable, I moved to the main business problem: where is value leaking? Churn risk matters, but it is not enough on its own. A high-risk customer only becomes a business priority when that risk is connected to revenue, margin, and long-term value. To measure that exposure, I combined annual revenue run-rate, margin run-rate, profit-adjusted CLV, future churned CLV, and leakage concentration across value deciles and behavioral segments.

The churn model came next. Its purpose was not only to predict which customers might leave, but to rank customers in a way that could support retention strategy. I also trained a sensitivity model without direct cancellation signals to test whether the ranking still held when obvious churn indicators were removed. This helped separate model performance from business usefulness.

I then added lifecycle context because two customers with the same churn probability may not deserve the same action. A long-tenure, high-engagement, high-value customer should be treated differently from a lower-value customer with similar risk. The lifecycle layer is therefore used as a risk-context proxy based on observed customer-month coverage, not as an exact event-time survival estimate.

Once the risk and value layers were built, I converted the model outputs into save-worthiness logic. This step moved the project from prediction to economics. The engine compared churn probability, CLV exposure, expected response assumptions, intervention cost, and expected net save value to separate customers who were simply at risk from customers who were economically worth a paid retention offer.

The final layer made the strategy testable. I used ROI simulation to compare budget levels, identify the efficient frontier, and estimate break-even response rates. Then I designed an A/B test plan so the campaign would not be scaled from model scores alone. The final recommendation only scales if churn improves, incremental net value is positive, and customer-experience guardrails hold.

## The business problem

Before building the model, I wanted to take a closer look at what kind of churn problem this actually was.

There is a big difference between churn that is spread across the whole customer base and churn that is concentrated in a few economically important groups. If risk is broad and shallow, a wide retention program might make sense. If value at risk is concentrated, the decision becomes more focused. Leadership would need to know which customers to protect first.

After scoring 970,960 customers, the size of the portfolio was clear. The customer base represented about $1.69B in annual revenue run-rate proxy, $1.10B in annual margin run-rate proxy and $974.4M in total profit-adjusted CLV proxy. Within that base, future churned CLV proxy was $137.7M, which implied a 14.13% CLV leakage rate.

At first, that leakage number looked like the headline.

It was not.

The more important finding was concentration. The top 10% of customers by CLV represented 23.89% of total CLV, but they accounted for 79.63% of future churned CLV proxy. The top 5% alone accounted for 78.09% of future churned CLV proxy.

That imbalance changed the direction of the project.

At that point, I stopped thinking about retention as a volume problem. The goal was not to contact as many customers as possible or build the largest possible campaign list. My main objective was to understand where churn could create the most financial damage, then decide where intervention would actually be worth the cost.

That made the project feel less like a marketing exercise and more like a capital allocation problem.

A broad campaign might look productive on paper, but the analysis suggested that the real opportunity was narrower. The business did not need to save every risky customer. It needed to identify the customers where churn risk, customer value, and intervention economics came together.

## Early revenue leakage signal

One segment made the risk feel operational almost immediately:

**Long tenure / high engagement / high revenue**

This group included 24,676 customers, carried a 43.39% churn risk, and represented $10.9M in future churned revenue proxy.

That combination stood out. It was not an obvious throwaway segment. Customers in this group had history, usage, and revenue behind them. They had already shown enough commitment to the product that losing them would raise a different question than losing low-activity users.

For me, this was where the analysis became more practical.

A long-tenure, high-engagement, high-revenue group with material churn risk points to something more serious than acquisition quality or casual customer drop-off. Valuable customers may be reaching moments where intervention needs to be more thoughtful.

Additional segment cuts supported the same concern. The protect high-value customer group represented 159,813 customers and $106.1M in future churned CLV proxy. The premium save budget tier represented 194,192 customers and $120.8M in future churned CLV proxy.

I used this finding as the bridge into the rest of the project.

A churn model could rank risk, but this segment showed why ranking alone would not be enough. From there, the next step had to be economic. I needed to understand which at-risk customers had enough value to justify attention, which customers should be suppressed, and which strategy would still make sense after cost, response, and experiment design were considered.

## Data foundation

Before modeling, I built a customer-level and customer-month analytical base so the later strategy work would rest on a clean foundation.

By the time the scoring table was finished, it contained 970,960 customers with one row per customer. Validation checks showed 0 duplicate customers, 0 null CLV values, and 0 negative value metrics. Every customer was also assigned to a value tier, action group, and budget tier.

That foundation mattered. ROI simulation and A/B test planning only work when the customer base underneath them is stable. A retention strategy can look precise on the surface, but its credibility still depends on the table it is built from.

## CLV and value exposure

The CLV layer estimates profit-adjusted customer value and separates customers into value-based groups.

The portfolio view produced:

```text
$974.36M in profit-adjusted CLV proxy
$137.70M in future churned CLV proxy
14.13% CLV leakage rate
10 CLV deciles
68 leakage segments
68 driver-score segments
```

The elite value tier contained 194,192 customers. This is the customer group where retention economics can become meaningful, but even here the answer is not automatic. High value still has to be combined with churn risk, response likelihood, and intervention cost.

That is why the project moves into churn modeling next.

## Churn model

For the churn layer, I used HistGradientBoosting as the strongest performing model.

The model reached 0.9808 ROC AUC and 0.9151 PR AUC. More importantly for a retention use case, the ranking quality was strong. The top risk decile produced an 8.86x lift, captured 88.58% of churners, and captured 92.20% of future churned CLV proxy.

I did not want to stop at the headline performance metrics.

A model can look excellent if it is mostly learning direct cancellation behavior. That kind of signal can still be useful, especially for short-term intervention, but it does not fully answer whether the model is learning broader patterns in customer behavior. I wanted to test that directly rather than assume it.

Thus, I trained a second version of the model with direct cancellation signals removed.

That sensitivity model still reached 0.9406 ROC AUC, 0.8186 PR AUC, and 7.75x top-decile lift. The performance dropped, as expected, but the model still ranked risk well. That gave me more confidence that the risk signal was not coming only from obvious cancellation indicators. Revenue behavior, payment patterns, engagement, and lifecycle features were still carrying meaningful information.

I treated this step as model due diligence, not just a performance check.

The final validation layer included feature importance review, leakage checks, calibration diagnostics, and business lift analysis. My goal was to understand whether the model could support prioritization under real retention constraints, not just whether it could produce a high score on a test set.

## Lifecycle context
After the churn and CLV layers, I wanted to add one more dimension: timing.

Risk and value are useful, but they do not fully capture the entirety of a customer relationship. A newer customer showing risk may require a different response from a long-tenure customer whose engagement is starting to shift. Lifecycle context helped me account for where a customer sits in that relationship, rather than relying only on a score at one point in time.

I also treated this layer carefully. In this dataset, the churn label is a future churn outcome, not a clean month-by-month event history. For that reason, I interpreted the survival-style outputs as lifecycle-risk proxies, not exact churn event-time estimates.

That distinction matters because it keeps the analysis from claiming more precision than the data can support.

Across 970,960 customers, the lifecycle layer identified 179,814 high-risk, high-value customers. Together, they represented $126.9M in future churned CLV proxy.

The strongest lifecycle priority segment was:

**Long tenure / high engagement / high revenue / elite value / critical risk**

That segment clarified why timing belonged in the strategy. These customers were not only valuable and risky. They were also far enough into the relationship that a behavioral shift would deserve closer attention before choosing a retention action.

By this point, risk, value, and lifecycle context were in place. Next came the economic layer: translating those signals into a save-worthiness score.

## Save-worthiness

This is where I wanted to push the project beyond a simple churn model.

I scored save-worthiness using expected economic value. The logic is:

```text
gross value at risk proxy =
predicted churn probability × profit-adjusted CLV proxy
expected saved CLV proxy =
gross value at risk proxy × expected intervention response rate
net save-value proxy =
expected saved CLV proxy − intervention cost proxy
```

I did not multiply by margin again because the CLV proxy already reflects the margin assumption. That kept the calculation from double-counting the same adjustment. The save-worthiness layer reduced the full population from 970,960 customers to 71,437 economically positive paid-offer candidates. That is only 7.36% of the customer base.

Under the base planning assumptions, those candidates represented $16.62M in expected saved CLV proxy against $791K in estimated intervention cost proxy. That produced a $15.83M expected net save-value proxy and a 20.00x paid-offer ROI proxy.

I treat those figures as scenario-planning outputs, not guaranteed campaign results. The actual return would still depend on response rate, offer redemption, customer behavior, and incremental lift against a holdout group.

This is the strongest decision point in the project because it turns risk and value into an economic filter.

The model does not say “save everyone at risk.” It identifies the customers who clear the expected-value threshold under the stated assumptions.

It also suppresses customers who should not receive paid offers. That part matters. In real retention work, not spending is also a decision.

## ROI simulation

After the save-worthiness layer, I used the ROI simulator to test how durable the campaign economics were.

A single base-case estimate was not enough for me. Retention planning depends on response rates, offer cost, targeting depth, and budget limits. Small changes in those assumptions can turn a strategy that looks attractive on paper into one that is much harder to defend in a business review.

The strongest base-case strategy was:

**All positive ROI paid-offer candidates**

Under the base planning assumptions, this strategy targeted 71,437 customers and saved an expected 4,705 customers. It produced $16.62M in expected saved CLV proxy against roughly $791K in campaign cost proxy, resulting in a $15.83M expected net save-value proxy. The modeled ROI proxy was 20.00x, with a 0.73% break-even response rate.

I treated those figures as planning estimates, not proof of live campaign performance.

What mattered more was the shape of the budget frontier. A $1.0M budget cap produced the highest absolute expected net value in the tested range. A $750K budget cap, however, captured 99.99% of the maximum tested net value. The extra spend from $750K to $1.0M added only about $2.3K in incremental net value.

That changed the recommendation.

Instead of arguing for the largest tested budget, I would recommend operating near the $750K efficient frontier. At that point, the campaign captures almost all of the modeled value while avoiding unnecessary capital commitment.

That is the tradeoff I would want leadership to see. The simulator does not only estimate whether the campaign can be profitable. It shows where additional spend stops producing meaningful return.

## A/B test planner

The final layer translated the modeled strategy into an experiment the business could actually evaluate.

ROI simulation can estimate expected value, but it cannot prove incremental lift. A high modeled return still depends on assumptions about response, redemption, behavior, and timing. Without a holdout group, the business would not know how much of the observed retention came from the offer instead of customers who would have stayed anyway.

For that reason, I designed the next step as a controlled save-worthy retention offer test.

The test used 62,836 candidate customers, split into 31,430 treatment customers and 31,406 control customers. It was built around the $750K efficient budget cap, which had already captured 99.99% of the maximum tested net value in the ROI simulation.

Under the planning assumptions, the treatment arm was expected to generate $7.93M in net save-value proxy against about $375K in campaign cost proxy. That produced a 21.15x treatment ROI proxy and a 0.61% break-even response rate.

Expected churn movement was also defined before launch. The planning control churn rate was 64.81%, compared with a planning treatment churn rate of 57.40%. That difference implies a 7.41 percentage point absolute churn reduction and an 11.43% relative churn reduction.

To make the test launch-ready, I added deterministic stratified randomization, treatment-control balance diagnostics, minimum detectable effect analysis, metric definitions, rollout decision rules, and an experiment risk register.

The decision rule was intentionally strict:

Do not scale from churn reduction alone.

Scaling would require three conditions: lower churn, positive incremental net value, and acceptable customer-experience guardrails. If the economics failed, the campaign should be killed or redesigned, even if the model ranking looked strong.

That is the standard I wanted the final project to meet. The model identifies the opportunity. The simulator sizes the opportunity. The experiment decides whether the business should trust it.

## Final recommendation

I would not recommend a blanket retention campaign.

A stronger path is to target the save-worthy paid-offer population, plan around the $750K efficient budget frontier, launch a controlled A/B test, and scale only if the experiment proves incremental net value.

By the end, the system answers the full business chain:

Where is revenue leaking?
Who is likely to churn?
Which customers are valuable?
Which customers are worth saving?
What budget should leadership test?
What ROI should we expect?
How do we prove the strategy worked?

## Why this project is different

Many churn projects stop after model evaluation.

This one keeps going.

It connects prediction to business action and includes the parts that are easy to ignore in a cleaner academic exercise: feature leakage risk, cancellation-signal sensitivity, budget tradeoffs, suppression logic, break-even response rates, treatment-control balance, and rollout guardrails.

The project is not trying to prove that a model is impressive.

It is trying to show that a retention decision can be made responsibly.

## Main files

The project pipeline is organized as follows.
```text
00_clean_data.py
01_build_customer_month_table.py
01b_validate_customer_month_table.py
02_sql_cohort_retention_analysis.py
03_clv_profitability_model.py
04_revenue_leakage_analysis.py
05_churn_model.py
06_survival_lifecycle_analysis.py
07_model_validation_business_impact.py
08_save_worthiness_scoring.py
09_retention_roi_simulation.py
10_ab_test_planner.py
```

Each script writes business-ready outputs into data/processed/, with SQL references saved in sql/.

## Tools used

Python, SQL logic, pandas, NumPy, scikit-learn, parquet outputs, model validation tables, and Tableau-ready datasets.

The project was built to be explainable. The final outputs are meant for a business audience, not just a notebook audience.

## Planned Part 2

The next version of this project will turn the retention strategy into an interactive Streamlit app.

The app will let a user adjust budget caps, response-rate assumptions, offer costs, and targeting strategies, then compare expected saved CLV proxy, campaign cost proxy, net save-value proxy, ROI proxy, and break-even response rate.

For this version, I focused on building the analytical engine, validating the decision logic, and documenting the business case.

## Interview summary

This project is my end-to-end customer retention strategy engine.

I built the data model, quantified revenue leakage, trained and validated a churn model, estimated profit-adjusted CLV, added lifecycle context, created save-worthiness scoring, simulated ROI, and designed an A/B test plan.

The part I would emphasize most is this:

A churn model is not enough. The real business question is whether the customer is worth saving, whether the campaign clears cost, and whether an experiment can prove incremental value.

That is the question this project answers.
