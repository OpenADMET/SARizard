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
finetune → analyze). Read `plots/prescaling_ranking_r2.csv` and the ablation report
card to pick the recipe, then harden it into the core workflow (TODO milestone 5).

## 250K training collapse and the corpus/regime redo

The first full triage (250K corpus, single seed, all three MPNN-LR protocols) completed 504
finetune runs, but every one of the 7 pretraining runs behind them had already diverged:
val/R2 and val_loss blew up by 2-6 orders of magnitude within a single epoch (`minimal` at
epoch 4, `chemeleon_baseline` at epoch 10; the rest in between), and the "best checkpoint"
saved by early stopping was whatever existed right before the blowup. The apparent recipe
ranking from that run was therefore dominated by which recipe's trajectory happened to
survive longest before collapsing, not by prescaling quality.

Auditing `../foundation-models/pretraining/run_pretraining.py`, which trains the same
MPNN/descriptor-regression task without this instability, found no gradient clipping in
either implementation, but three real regime departures: `PATIENCE` 5 vs. the sibling's 50,
`FNN_HIDDEN_SIZE`/predictor width 2048 vs. 1024, and the masked-pretext keep fraction 70%
(`DROPOUT_FRACTION=0.30`) vs. the sibling's 15% (`MASKING_RATIO=0.15`), a much denser
per-step supervision load on a 3585-dim target block. The sibling also trains bf16-mixed
precision where this repo trained full fp32.

The sibling's regime is now adopted as canonical in `sarizard/pretraining/config.py`
(`PATIENCE=50`, `FNN_HIDDEN_SIZE=1024`, `WARMUP_EPOCHS=2`, `DROPOUT_FRACTION=0.85`), with
`GRADIENT_CLIP_VAL=0.5` and bf16/16-mixed precision added on top, since neither
implementation had those. The triage is being rerun on the full corpus
(`corpus/corpus_full.parquet`, ~900K molecules vs. the 250K screening subset) with this
fixed regime, one recipe (`chemeleon_baseline`) first to confirm stability before firing the
other six. The original 250K runs (targets, splits, foundations, configs, results, plots) are
archived at `archive/ablation_250k_pre_gradclip/` for reference; see `FINDINGS.md` for the
numbers, which are superseded and should not be used to pick the production recipe.

## MPNN learning-rate sweep

The triage originally finetuned every prescaling recipe frozen only (`mpnn_lr=0`), which ranks
the preprocessing by representation quality alone. To check that the ranking survives once the
backbone can adapt, the same ablation foundations are also finetuned under the two
[[Finetune Protocols#Learning-rate experiments|LR protocols]]: `reduced` (`mpnn_lr=1e-4`) and
`unlocked` (`mpnn_lr=1e-3`). This crosses the prescaling axis with the finetune axis, so a
recipe that wins frozen but loses once the MPNN moves is caught before it is baked in.

The recipes are generated with `sarizard/configs/generate.py --mpnn-lr-mode {reduced,unlocked}`
into `configs/ablation_<name>__s42__{reduced,unlocked}/` and finetuned through
`ablation_finetune.sbatch` in the `openadmet` env, alongside the frozen
`configs/ablation_<name>__s42/`. `prescaling_report` is protocol-aware: it builds a report card
and ranking per protocol (frozen keeps the unsuffixed filenames, the others add a `_<mode>`
suffix) and, when more than one protocol is present, writes
`plots/prescaling_mode_comparison_<metric>.csv` (each recipe's mean metric under frozen,
reduced, and unlocked side by side). If a recipe wins frozen but loses once the backbone can
move, that comparison catches it before the recipe is baked in.

## Related

- Feeds [[Shared Corpus and Regime]] (the prescaling becomes part of the fixed regime)
- Uses the [[osmordred]] target as the representative continuous flavor
- Read against the [[Report Card]] format (endpoints by ablation instead of by flavor)
