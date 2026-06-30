---
tags: [method, status/blue]
---
# Meta-Model

> **Summary:** The second north-star artifact. For each endpoint it stacks the per-flavor
> finetuned predictions into a small model (LGBM, random forest, or MLP) and compares its
> cross-validated score to the best single flavor. It answers: does an ensemble of foundations
> beat the best single foundation.

## How it is built

- `analysis/meta_model.py` reads the cached per-flavor test predictions (the `y_pred.npy`
  files written by [[Report Card]]'s evaluate step) and, per endpoint, builds a feature matrix
  of one column per flavor.
- The stacker is cross-validated over the test molecules, so its reported score is
  out-of-fold and never trained on the rows it scores. A single flavor's prediction is a fixed
  vector, so its direct test score is already honest.
- Output: a per-endpoint CSV of meta vs best-single R², a lift bar chart, and a summary of how
  often the ensemble wins.

## Leakage discipline

The flavor models never trained on the test molecules, and the meta-model is the only thing
fit here, so the cross-validation makes the comparison leakage-safe. The ensemble pays a CV
cost the single flavors do not, so a positive lift is conservative.

## Related

- Consumes predictions cached by [[Report Card]].
- Stacks all flavors; see the flavor nodes under [[_index]].
