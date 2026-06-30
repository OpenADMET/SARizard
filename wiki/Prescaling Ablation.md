---
tags: [method, status/blue]
---
# Prescaling Ablation

> **Summary:** A triage run before the flavor sweep to decide how continuous descriptor
> targets are preprocessed before pretraining. One representative flavor (osmordred) is driven
> through several prescaling recipes with the backbone, corpus, and regime held fixed; the
> recipe that transfers best downstream is baked into [[Shared Corpus and Regime]] and applied
> identically to every continuous flavor.

## Why first

Descriptor blocks carry NaNs and infs, heavy-tailed outliers, redundant columns, and
near-constant columns. How those are handled changes the pretraining target, and a target
artifact would confound every column of the [[Report Card]]. Fixing the preprocessing once,
up front, keeps the flavor comparison clean.

The current production path (`split.py`) computes mean/std on the raw target and uses those
same stats both to set the winsorization limits (mean ± 6·std) and to z-score, so the
outliers it is about to clip inflate the std first. The triage measures whether correcting
that order and adding steps actually helps.

## The steps (in canonical order)

1. drop invalid (mandatory): drop columns above a NaN/inf fraction, then replace NaN→0,
   +inf→column max, -inf→column min. Cleaning is always applied; training breaks on inf.
2. winsorize: clip each column to a robust range (percentile- or std-based).
3. drop correlated: drop the later column of each pair with sampled |r| > 0.98.
4. Yeo-Johnson: per-column power transform toward normality.
5. drop low variance: drop near-constant columns.
6. z-score: subtract mean, divide by std.

All fitting happens on the train chunks only (no leakage from the validation rows), then the
column drops and transforms are applied to every row.

## Ablation ladder

- `minimal` — clean + z-score only (floor: does winsorization even help?)
- `chemeleon_baseline` — reproduces today's `split.py` (std winsorize + raw-stat z-score, entangled order)
- `order_fix` — winsorize (percentile) first, then z-score on the winsorized data
- `plus_drop_corr` — `order_fix` + correlated-column drop
- `plus_drop_low_var` — `order_fix` + low-variance drop
- `plus_yeo_johnson` — `order_fix` + Yeo-Johnson
- `full` — all steps stacked

Each `plus_*` isolates one step's marginal effect over `order_fix`; `full` stacks them.

## Run it

`bash slurm/run_ablations.sh` submits the chain (corpus → target → prescale → pretrain →
finetune → analyze). Read `analysis/plots/prescaling_ranking_r2.csv` and the ablation report
card to pick the recipe, then harden it into the core workflow (TODO milestone 5).

## Related

- Feeds [[Shared Corpus and Regime]] (the prescaling becomes part of the fixed regime)
- Uses the [[osmordred]] target as the representative continuous flavor
- Read against the [[Report Card]] format (endpoints by ablation instead of by flavor)
