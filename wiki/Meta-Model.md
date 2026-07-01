---
tags: [method, status/blue]
---
# Meta-Model

> **Summary:** The second north-star artifact. For each endpoint it stacks the per-flavor
> finetuned predictions into a small model (LGBM, random forest, or MLP) and compares its
> cross-validated score to the best single flavor. It answers: does an ensemble of foundations
> beat the best single foundation.

## How it is built

- `sarizard/analysis/meta_model.py` reads the cached per-flavor test predictions (the `y_pred.npy`
  files written by [[Report Card]]'s evaluate step) and, per endpoint, builds a feature matrix
  of one column per flavor. A flavor's seed variants (see [[Finetune Protocols]]) are averaged
  into that one feature.
- The stacker is cross-validated over the test molecules, so its reported score is out-of-fold
  and never trained on the rows it scores. The best-single-flavor baseline is scored the same
  way: the winning flavor is chosen on each fold's training rows and scored on its held-out
  rows, so selecting the baseline never peeks at the labels the stacker is graded on.
- Output: a per-endpoint CSV of meta vs best-single R², a lift bar chart, and a summary of how
  often the ensemble wins.

## Leakage discipline

The flavor models never trained on the test molecules, and the meta-model is the only thing
fit here, so the cross-validation makes the comparison leakage-safe. Because the best-single
baseline is also selected fold-wise (not by peeking at the full test set), both sides of the
lift are out-of-fold; the ensemble pays a CV cost the fixed single-flavor vectors do not, so a
positive lift is conservative.

## Related

- Consumes predictions cached by [[Report Card]].
- Stacks all flavors; see the flavor nodes under [[_index]].
