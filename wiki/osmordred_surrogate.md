---
tags: [control, status/running]
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

## Reading the result

- Lands near **0.327** (sweep osmordred): the surrogate flavor's strength was the ADME target
  itself, since swapping the target to descriptors on the same corpus loses the lead.
- Lands near **0.369** (surrogate_adme): the strength was the Novartis chemical space, since the
  lead survives replacing the ADME target with descriptors.

## How it runs

Frozen protocol, 5 finetune seeds (1-5) off one seed-42 foundation, matching every other flavor's
5-seed legs and tested against the 5-seed stock baseline. The driver
`slurm/run_osmordred_surrogate.sh` runs it end to end: target on the Novartis corpus, split,
pretrain (seed 42), batched 5-seed finetune, then `slurm/osmordred_surrogate_analyze.sbatch`
evaluates into a dedicated `results/osmordred_surrogate_metrics.csv` (never the shared
`results/metrics.csv`) and prints the comparison via `sarizard/analysis/control_report.py` (mean
R² ± seed std per condition, a Welch test vs the baseline, and which context arm it lands nearest).

## Status

Running. The first driver (2026-07-23) built the foundation and finished finetune batch 1
(50 of 120 recipes) but died submitting batch 2 with a Slurm "Job dependency problem": the batched
finetune re-applied the `afterok:<pretrain>` gate to every batch, and by batch 2 the pretrain job
had completed and aged out of Slurm's records. Fixed in `slurm/submit_batched.sh` to gate only the
first submission, then relaunched off the existing foundation to finish the remaining recipes and
the analyze step. `TODO.md` carries the full account; the result is not yet written up here or in
`FINDINGS.md`.

## Related

- Target twin: [[osmordred]] · Corpus twin: [[surrogate_adme]]
- Regime it deviates from (corpus only): [[Shared Corpus and Regime]]
- Interprets a row on the [[Report Card]] without appearing on it
