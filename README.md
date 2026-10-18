# Customer Churn Analysis — telecom

I did churn analysis early in my career and it stuck with me — it's one of
those problems where the statistics and the business story line up neatly.
This is a portfolio rebuild of that kind of work on **synthetic data**
(4,000 fictional telecom customers, generated in `analysis.py`).

## What I found

- **Contract type dominates.** Month-to-month customers churn at a far higher
  rate than one- or two-year contract holders — the model's largest
  coefficient by a mile (`contract_Two year: -0.84`).
- **Tenure is the second signal** (`-0.80`): the longer someone stays, the
  less likely they are to leave. The tenure distribution chart shows it
  visually — churners cluster hard on the left.
- **Support calls are a leading indicator** (`+0.54`). Customers who call
  support repeatedly aren't engaged, they're frustrated — each additional
  call meaningfully raises churn odds.
- Monthly charges matter, but less than you'd think (`+0.31`). Price isn't
  the story here; commitment and experience are.

The logistic regression scores **AUC 0.809** — solid for a four-feature model,
and deliberately simple. For a retention campaign, I'd rather hand the
business a model everyone understands than squeeze out two more AUC points
with something nobody can explain.

## Run it

```bash
pip install -r requirements.txt
python analysis.py
```

The script prints the model results and saves three charts into `images/`:
churn rate by contract type, tenure distributions (churned vs stayed), and
feature impact. The headline findings are above — the charts are worth a
look if you want the visual version.

## If this were production

Score the base nightly, push the top-risk decile to the retention team's
queue, and A/B test the save offer. The model is the easy part — the
hard part is getting the intervention in front of the right customer
before they port their number out.
