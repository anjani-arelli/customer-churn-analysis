# Model notes — why this model, and what it can't tell you

`analysis.py` fits a logistic regression on four features. That choice is
deliberate, and the trade-offs are worth writing down, because the model
someone can explain to a retention manager beats the model that scores
two points higher and nobody trusts.

## Why logistic regression

- **The coefficients are the story.** Contract type (`-0.84` for a two-year
  contract), tenure (`-0.80`), support calls (`+0.54`), monthly charges
  (`+0.31`) — each number answers a business question directly: "what
  moves churn, and by how much?" A gradient-boosted model would need a
  SHAP plot to say the same thing, and a SHAP plot doesn't survive a
  meeting with a non-technical audience.
- **The data is synthetic and small (4,000 rows).** Four features, one
  clear signal structure. Anything fancier would mostly be fitting
  noise and calling it insight.
- **AUC 0.809** on the held-out test set. Solid for a four-feature model.
  Good enough to rank customers for a retention queue; not good enough
  to auto-cancel anyone's contract, which nobody should be doing
  anyway.

## What each feature is really measuring

- **Contract type** is commitment, not a product feature. Month-to-month
  customers can leave tomorrow; a two-year contract holder has already
  decided to stay. The model is picking up intent, and that's why it's
  the largest coefficient.
- **Tenure** is survivorship. Customers who were going to churn mostly
  already did. The remaining long-tenure base is self-selected loyal —
  so tenure predicts staying, but it doesn't *cause* it. Don't build a
  campaign around "make people older."
- **Support calls** are the most actionable signal. A customer calling
  support repeatedly isn't engaged, they're frustrated — and the data
  agrees, with a positive coefficient. This is the one I'd hand to
  operations first: flag the third call in a month, not the tenth.
- **Monthly charges** matter less than you'd think (`+0.31`). Price
  isn't the story here; commitment and experience are. That cuts
  against the obvious "offer a discount" reflex — a discount to a
  frustrated month-to-month customer buys a month, not loyalty.

## Thresholds, and the default I'd change

The script reports results at the default 0.5 probability threshold,
which is fine for the AUC comparison but wrong for the actual job.
Churn is the minority class, and missing a churner costs more than
flagging a safe customer: a retention call to someone who was staying
is a minor annoyance; a churner nobody called is lost revenue.

In production I'd score the base nightly, rank by predicted
probability, and work down the top-risk decile until the retention
team's capacity runs out — no hard threshold at all. If a threshold
were required, I'd pick it from the precision-recall curve at the
point where the team can actually act on every flagged account,
not from a textbook default.

## What the model misses

- **No interaction terms.** The real risk is probably concentrated
  where signals combine — month-to-month *and* high support calls —
  and a main-effects model averages that away.
- **No time dimension.** Tenure is a snapshot, not a trajectory. A
  customer whose support calls jumped from zero to four this month
  looks the same as one who always called four times. Recency of
  behaviour would be the first feature I'd add with real data.
- **No usage data.** Minutes, data, roaming — the things that
  actually describe the product experience — aren't in this synthetic
  set. With real data they'd likely matter more than monthly charges.
- **Calibration is untested.** The model ranks well (that's what AUC
  measures), but I haven't checked whether "predicted 30%" actually
  means 30 out of 100 churn. Before anyone quoted a probability to
  the business, I'd want a calibration curve on the test set.

## If this were production

Score nightly, push the top-risk decile to the retention queue, A/B
test the save offer, and re-check calibration monthly. The model is
the easy part — the hard part is getting the intervention in front
of the right customer before they port their number out.
