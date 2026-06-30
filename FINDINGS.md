# Findings: fit-to-purpose CheMeleon foundations

Source of record for headline results. Per-flavor and per-endpoint detail lives in the
Obsidian wiki under `wiki/`; where the wiki and this file disagree, this file wins.

## The question

Does the descriptor target a CheMeleon-style foundation is pretrained against determine
which ADMET endpoints and endpoint families it serves best? And does stacking the
per-flavor finetuned predictions into a meta-model beat the best single foundation?

## Status

Scaffolding. No training results yet. The first result expected is the prescaling decision
(below), which precedes and gates the flavor sweep; the report card follows once the first
flavors finish.

## Prescaling

To be filled. Before the flavor sweep, one descriptor-preprocessing recipe is chosen by the
ablation triage (`pretraining/prescaling.py`, `slurm/run_ablations.sh`): osmordred is driven
through each recipe (`minimal`, `chemeleon_baseline`, `order_fix`, `plus_drop_corr`,
`plus_drop_low_var`, `plus_yeo_johnson`, `full`) with the backbone, corpus, and regime fixed,
so the only difference is the prescaling. The read to capture here: which recipe wins on mean
downstream R-squared and endpoint wins (`analysis/plots/prescaling_ranking_r2.csv`), the
margin over the `chemeleon_baseline` reproduction of today's `split.py`, and whether the
order fix alone (winsorize before z-score) accounts for most of the gain. The winning recipe
is then baked into the core workflow and applied identically to every continuous flavor.

## Report card

To be filled. The artifact is `analysis/report_card.py`: rows are endpoints across all
benchmark sets, columns are foundation flavors, each cell is one selectable metric
(default R-squared). The read to capture here: which flavor wins each endpoint family,
and whether any flavor dominates or is dominated overall.

## Per-flavor read

To be filled, one short paragraph per flavor as results land. Expected priors to test:

- Continuous descriptor flavors (osmordred, rdkit2d, erg) should be the strongest general foundations.
- Binary fingerprint flavors (ecfp, atompair, pubchem, e3fp) are leaky/weak pretexts and
  may underperform; confirm or refute.
- 3D flavors (usrcat, whim, e3fp) encode information absent from the 2D graph; test
  whether that helps any endpoint family.
- Learned-model flavors (minimol, surrogate_adme, ml_qm) distill another model's
  knowledge; test whether that transfers better than hand-crafted descriptor targets.

## Meta-model

To be filled. Whether the stacked ensemble of foundations beats the best single flavor
per endpoint, and which flavors carry the ensemble.
