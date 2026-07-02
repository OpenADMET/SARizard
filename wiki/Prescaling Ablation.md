---
tags: [method, status/green]
---
# Prescaling Ablation

> **Summary:** A triage run before the flavor sweep to decide how continuous descriptor
> targets are preprocessed before pretraining. One representative flavor (osmordred) is driven
> through several prescaling recipes with the backbone, corpus, and regime held fixed; the
> recipe that transfers best downstream is baked into [[Shared Corpus and Regime]] and applied
> identically to every continuous flavor.
>
> **Result:** complete, including the frozen/reduced/unlocked cross-protocol check.
> `chemeleon_baseline` wins by mean R-squared under the frozen protocol (the one the flavor
> sweep uses) on the full corpus with the regime fix, and since `split.py` already reproduces
> `chemeleon_baseline`, no code change was needed to bake it in. The margin over the
> runner-up, `order_fix`, is narrow, not decisive, once all three protocols are considered.
> A follow-up check on the 250K screening corpus found a *different* ranking altogether
> (`chemeleon_baseline` drops to 5th of 7 under frozen there), so the recipe ranking is
> corpus-size sensitive; see the 250K section below and [[Shared Corpus and Regime]] for why
> that matters. See `FINDINGS.md` for the full ranking.

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
implementation had those. The triage was rerun on the full corpus
(`corpus/corpus_full.parquet`, 944296 molecules vs. the 250K screening subset) with this
fixed regime: `chemeleon_baseline` first, alone, to confirm stability (clean through 15
epochs, `val/r2` 0.773 to 0.952), then the other six recipes, all of which also completed
clean (full 100 epochs, stable losses, no collapse). The original 250K runs (targets,
splits, foundations, configs, results, plots) are archived at
`archive/ablation_250k_pre_gradclip/` for reference; see `FINDINGS.md` for those numbers,
which are superseded and were not used to pick the production recipe.

Finetuning (168 runs: 7 recipes x 24 endpoints) hit one unrelated infrastructure gap: the
`openadmet` env's `boto3`/`botocore` versions were mismatched, breaking `openadmet anvil`'s
CLI import before any recipe ran. Fixed by reinstalling matching versions; the rerun
completed all 168 finetunes and the chained analyze step clean.

**Full-corpus result:** `chemeleon_baseline` wins by mean R-squared under the frozen
protocol (0.352), ahead of `order_fix` (0.342), `plus_yeo_johnson` (0.340),
`plus_drop_low_var` (0.327), `full` (0.326), `plus_drop_corr` (0.324), and `minimal` (0.307,
worst). Since `split.py` already reproduces `chemeleon_baseline`, Milestone 5 (bake the
winning recipe into the core workflow) needed no code change. See `FINDINGS.md` for the full
read.

## MPNN learning-rate sweep

The triage originally finetuned every prescaling recipe frozen only (`mpnn_lr=0`), which ranks
the preprocessing by representation quality alone. To check that the ranking survives once the
backbone can adapt, the same ablation foundations were also finetuned under the two
[[Finetune Protocols#Learning-rate experiments|LR protocols]]: `reduced` (`mpnn_lr=1e-4`) and
`unlocked` (`mpnn_lr=1e-3`). This crosses the prescaling axis with the finetune axis, so a
recipe that wins frozen but loses once the MPNN moves is caught before it is baked in.

The recipes were generated with `sarizard/configs/generate.py --mpnn-lr-mode {reduced,unlocked}`
into `configs/ablation_<name>__s42__{reduced,unlocked}/` and finetuned through
`ablation_finetune.sbatch` in the `openadmet` env (job 18443536, 504 runs, all completed),
alongside the frozen `configs/ablation_<name>__s42/`. `prescaling_report` is protocol-aware: it
builds a report card and ranking per protocol (frozen keeps the unsuffixed filenames, the
others add a `_<mode>` suffix) and, since more than one protocol is present, wrote
`plots/prescaling_mode_comparison_r2.csv` (each recipe's mean metric under frozen, reduced, and
unlocked side by side).

**Result: the winning recipe shifts by protocol.** `chemeleon_baseline` wins frozen (mean
R-squared 0.352), `order_fix` wins reduced (0.388), `plus_drop_corr` wins unlocked (0.322).
`reduced` is uniformly the strongest protocol for every recipe; `unlocked` compresses the
spread between recipes (0.28-0.32), so prescaling choice matters less once the backbone can
fully adapt, echoing the pre-fix 250K-era read even though those numbers were invalid.
`plus_drop_low_var` is the clear bottom performer in every protocol.

`chemeleon_baseline` and `order_fix` are close enough that a ranked-choice election (each
endpoint's R-squared ranking treated as a ballot, instant-runoff across all three protocols
pooled) came down to 44 votes to 43 in `order_fix`'s favor in the final round, despite
`chemeleon_baseline` leading every earlier round. This does not overturn the Milestone-5
decision (`chemeleon_baseline`, since frozen is the only protocol the flavor sweep actually
uses), but downgrades it from a clear win to a narrow one; see `FINDINGS.md` for the full
tables.

## 250K corpus-size check

The full-corpus triage above settled Milestone 5, but the flavor sweep it feeds
([[Shared Corpus and Regime]]) runs on the smaller 250K screening corpus, not the full
corpus the triage itself used. To check whether that matters, the same 7 recipes x 3
protocols triage was repeated on `corpus/corpus_250k.parquet`, with the regime fix in place
(unlike the original pre-fix 250K run, which is invalid and archived separately). The
full-corpus artifacts were archived to `archive/ablation_full_corpus/` first so the rerun
did not overwrite them, mirroring the existing archive precedent. All 504 finetunes and the
chained analyze completed clean.

**Result: the 250K ranking does not match the full-corpus ranking.** Winners are
`plus_drop_low_var` (frozen, mean R-squared 0.347), `plus_yeo_johnson` (reduced, 0.381), and
`minimal` (unlocked, 0.324), none of which match the full-corpus winners
(`chemeleon_baseline`, `order_fix`, `plus_drop_corr`). Most strikingly, `chemeleon_baseline`
(the full-corpus frozen winner and the Milestone-5 decision) drops to 5th of 7 under frozen
at 250K (0.295). Averaged across protocols, most recipes score higher on the full corpus as
expected, but `plus_drop_low_var` and `minimal` score lower, so it isn't simply "more data
helps everyone." `chemeleon_baseline` starts mediocre at 250K (4th of 7) and gains the most
of any near-top recipe going to the full corpus, so it reads as a recipe that scales well
rather than one that is intrinsically strong.

Does not change the Milestone-5 decision (the flavor sweep's protocol is frozen, and the
full-corpus frozen ranking is what that decision is based on), but it does mean the recipe
ranking should not be assumed to transfer to a different corpus size, including the 250K
corpus the flavor sweep itself uses. See `FINDINGS.md` for the full tables and the pooled
ranked-choice comparison.

## Related

- Feeds [[Shared Corpus and Regime]] (the prescaling becomes part of the fixed regime); note
  the corpus-size mismatch flagged there (the triage decision comes from the full corpus, the
  flavor sweep runs on 250K)
- Uses the [[osmordred]] target as the representative continuous flavor
- Read against the [[Report Card]] format (endpoints by ablation instead of by flavor)
