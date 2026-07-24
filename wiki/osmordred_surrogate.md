---
tags: [control, status/complete]
---
# osmordred_surrogate

> **Summary:** Standalone control that isolates [[surrogate_adme]]'s chemical space from its
> target. It computes the [[osmordred]] descriptor target (3585 dims, MSE) on `surrogate_adme`'s
> Novartis corpus, holding the target identical to sweep [[osmordred]] and the corpus identical to
> [[surrogate_adme]]. Where it lands separates the two: near sweep osmordred means the surrogate
> flavor's strength was its target, near surrogate_adme means it was the chemical space.

- Target: 3585 osmordred 2D descriptors (MSE), the same block sweep [[osmordred]] regresses
- Corpus: [[surrogate_adme]]'s 273,706-molecule Novartis set, not the [[Shared Corpus and Regime]]
- Standalone by construction: excluded from `flavor_names()`, so off the [[Report Card]], out of
  the flavor sweep, and out of the [[Meta-Model]]
- Calculator: reuses `sarizard/pretraining/features/osmordred_target.py` via the `calculator`
  field on the flavor registry; the corpus is selected through `corpus_from` pointing at
  `surrogate_adme`

## Why it exists

[[surrogate_adme]] leads frozen transfer (mean R² 0.369) but is confounded two ways against sweep
[[osmordred]] (0.327): a different corpus (Novartis molecules) and a different target (25 ADME
predictions vs 3585 osmordred descriptors). This control fixes the target to osmordred's block and
keeps surrogate_adme's corpus, so a single number attributes the surrogate lead. The Novartis
corpus is ~273K vs the sweep's 944K, a minor size confound; the earlier 250K-vs-full check showed
osmordred-family recipes gain only modestly with more data, so a small corpus that still scores
high strengthens the chemical-space read rather than weakening it.

## Result

Mean R² 0.325 ± 0.004 (5 seeds), against surrogate_adme 0.369, sweep osmordred 0.305, and stock
0.295. It lands nearest sweep osmordred (|Δ|=0.020, vs 0.044 to surrogate_adme) and clears stock
significantly (+0.031, Welch p<0.001). Holding surrogate_adme's Novartis corpus while swapping its
ADME target for the osmordred descriptor target drops transfer from 0.369 to 0.325, so
**surrogate_adme's lead was driven mostly by its on-task ADME target, not its chemical space**. The
Novartis space contributes a little (the control sits above both sweep osmordred and stock), but
the target dominates. The reading rubric this decided:

- Landed near sweep osmordred: the surrogate flavor's strength was the ADME target itself, since
  swapping the target to descriptors on the same corpus loses most of the lead.
- Would have landed near surrogate_adme: the strength was the Novartis chemical space, since the
  lead survives replacing the ADME target with descriptors. (Not what happened.)

## How it runs

Frozen protocol, 5 finetune seeds (1-5) off one seed-42 foundation, matching every other flavor's
5-seed legs and tested against the 5-seed stock baseline. The driver
`slurm/run_osmordred_surrogate.sh` runs it end to end: target on the Novartis corpus, split,
pretrain (seed 42), batched 5-seed finetune, then `slurm/osmordred_surrogate_analyze.sbatch`
evaluates into a dedicated `results/osmordred_surrogate_metrics.csv` (never the shared
`results/metrics.csv`) and prints the comparison via `sarizard/analysis/control_report.py` (mean
R² ± seed std per condition, a Welch test vs the baseline, and which context arm it lands nearest).

## Status

Complete (2026-07-24). The first driver (2026-07-23) built the foundation and finished finetune
batch 1 (50 of 120 recipes) but died submitting batch 2 with a Slurm "Job dependency problem": the
batched finetune re-applied the `afterok:<pretrain>` gate to every batch, and by batch 2 the
pretrain job had completed and aged out of Slurm's records. Fixed in `slurm/submit_batched.sh` to
gate only the first submission, then relaunched off the existing foundation; all 120 recipes
finished and analyze wrote `results/osmordred_surrogate_metrics.csv`. `TODO.md` carries the full
account.

## Related

- Target twin: [[osmordred]] · Corpus twin: [[surrogate_adme]]
- Regime it deviates from (corpus only): [[Shared Corpus and Regime]]
- Interprets a row on the [[Report Card]] without appearing on it
