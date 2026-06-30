---
tags: [method, status/blue]
---
# Report Card

> **Summary:** The first north-star artifact. A heatmap with endpoints as rows and foundation
> flavors as columns, each cell one selectable metric (default R²). Color is row-relative, so
> the best flavor for each endpoint is greenest regardless of the metric's absolute scale. It
> answers: which pretraining target serves which endpoint and endpoint family best.

## How it is built

- `analysis/evaluate.py` reloads each finetuned model, predicts on the held-out test split,
  and writes per-(flavor, endpoint) metrics to `results/metrics.csv` (caching predictions for
  the [[Meta-Model]]).
- `analysis/report_card.py --metric r2` pivots that CSV to an endpoints-by-flavors matrix and
  renders the heatmap plus a matrix CSV. Swap `--metric` for rmse, mae, spearman, kendall, rae.

## Reading it

- Rows group by dataset, then endpoint; columns are flavors in registry order.
- A green cell is the best flavor for that endpoint; recolor a flavor node by how it does here.
- Binary fingerprint flavors are a leaky, weak pretext; a poor column for them is a result.

## Related

- Built from results of every flavor finetuned over [[Clearance]], [[Permeability]],
  [[Solubility]], [[Lipophilicity]], [[Potency]], [[CYP Inhibition]], and [[hERG]].
- Paired artifact: [[Meta-Model]].
