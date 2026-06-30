# TODO: fit-to-purpose CheMeleon foundations

Working goal: determine whether the descriptor target a CheMeleon-style foundation is
pretrained against changes which downstream ADMET endpoints (and endpoint families) it
serves best, and whether an ensemble of foundations beats the best single one.

Core question: does a foundation pretrained on osmordred, RDKit descriptors,
fingerprints, pharmacophores, 3D descriptors, minimol embeddings, or surrogate ADME
predictions specialize toward particular endpoint families (clearance, permeability,
solubility, potency, CYP inhibition, hERG, PXR)?

Headline results and the read on each flavor: `FINDINGS.md`.

## Milestones (ranked)

- [ ] 1. Scaffold the repo, copy data, wire the conda environments. (in progress)
- [ ] 2. Prepare the shared 250K corpus from the original CheMeleon PubChem set (Zenodo
  DOI 10.5281/zenodo.15733574), single seed, persisted, with cached default-featurizer graphs.
- [ ] 3. Drive osmordred end to end as the validation flavor: compute target, pretrain
  (MeanAggregation, DEFAULT featurizer), convert checkpoint, finetune one endpoint,
  confirm the foundation loads and a sane R-squared lands. This validates the checkpoint
  bridge and the featurizer-dim match before any fan-out.
- [ ] 4. Prescaling ablation triage (runs BEFORE the flavor sweep). Drive osmordred through
  every prescaling recipe in `sarizard/pretraining/prescaling.py` (`chemeleon_baseline`, `order_fix`,
  `plus_drop_corr`, `plus_drop_low_var`, `plus_yeo_johnson`, `full`, and the `minimal` floor),
  pretrain and finetune from each, and compare downstream endpoint performance. Submit with
  `bash slurm/run_ablations.sh`; read `plots/prescaling_ranking_r2.csv` and the
  ablation report card to pick the production recipe. The backbone, corpus, and regime are
  fixed across ablations, so the difference is the prescaling alone.
- [ ] 5. (GATED on 4) Harden the chosen prescaling into the core flavor-sweep workflow. Wire
  the winning `PrescalingConfig` into the default `split.py` path (or insert a prescale step
  ahead of it) so every flavor pretrains on the same, vetted preprocessing. Until this lands,
  the flavor sweep keeps the current `chemeleon_baseline` behavior. Record the decision and
  the margin over baseline in `FINDINGS.md`.
- [ ] 6. Fan out the direct-compute flavors on the cluster: rdkit2d, erg, ecfp, atompair,
  pubchem, the 3D set (usrcat, whim, e3fp), and jazzy (isolated env for its RDKit pin).
- [ ] 7. Add the learned-model flavors: minimol, surrogate_adme, ml_qm. Each runs its
  source model over the shared corpus in an isolated environment and caches the target.
- [ ] 8. Report card: heatmap of endpoints by flavors with a selectable metric (default R-squared).
- [ ] 9. Meta-model: stack per-flavor finetuned predictions per endpoint, fit LGBM/RF/MLP
  on out-of-fold predictions, compare to the best single flavor.

## Open items (need input or external data)

- [x] ML-QM flavor: use qmdesc (Guan et al., MIT, bundled weights). It predicts 4 atom-level
  (partial charge, nucleophilic/electrophilic Fukui, NMR shielding) and 2 bond-level (bond
  order, bond length) QM descriptors. Pooled per descriptor with mean/std/min/max to a
  24-dim molecule target. Open design choice: a richer alternative is to regress the
  per-atom/per-bond descriptors directly as node/edge targets (chemprop supports this),
  which keeps the resolution that pooling discards; revisit if the pooled target underperforms.
- [ ] SLURM specifics: partition and account names, GPU type, per-job time limits, whether
  the conda envs exist on the cluster, and whether the cluster shares this filesystem.
- [ ] Surrogate-ADME data: download the Novartis Nat Commun 2024 released CSV
  (DOI 10.1038/s41467-024-49979-3, Supplementary Data 1, CC BY 4.0) and run
  `compute_target --flavor surrogate_adme --csv-path <path>`. The CSV is the pretraining
  corpus for this flavor; no model training step is needed.
- [ ] MLIP conformer backend for the 3D flavors (usrcat, whim, e3fp): the calculators
  currently use RDKit ETKDG + MMFF94 (seeded, in `features/skfp_targets.py`). Once the
  pipeline runs end to end, add an ML-potential backend (candidates: Auto3D with
  ANI2x/AIMNet2, or MACE-OFF23 via ASE/openmm-ml) in an isolated GPU env and make it the
  pluggable conformer source; compare descriptor stability against MMFF94.

## Future experiments

- [ ] Reduced MPNN LR: repeat the full finetuning sweep with `mpnn_lr` set to a fraction of
  `ffn_lr` (e.g. 1e-4 vs 1e-3) rather than 0. Tests whether partial unfreezing recovers
  performance on endpoints where the frozen backbone underperforms random init, or whether
  it simply reintroduces the initialization-washing problem.
- [ ] Fully unlocked MPNN: repeat with `mpnn_lr` equal to `ffn_lr` (1e-3). Establishes the
  upper bound on what full finetuning can achieve and quantifies how much signal the frozen
  protocol sacrifices; the gap between frozen and unlocked is the cost of the clean ablation.
- [ ] Target-dropout fraction for small flavors: the masked-pretext dropout in `losses.py`
  (`DROPOUT_FRACTION=0.30`, applied per target element to every flavor) keeps a fixed
  fraction, not a fixed count. Its rationale (stop the head co-adapting across a wide
  descriptor block) is strong at 3585 dims (osmordred) but weak at low dims: jazzy (6) keeps
  ~4 of 6 targets per step with high variance (Binomial(6, 0.7), std ≈ 1.1), so the dropout
  mostly injects gradient noise. Mechanically safe (loss aggregates over all kept elements in
  the batch, not per-row, so no divide-by-zero). If jazzy/ml_qm/surrogate_adme underperform,
  ablate the fraction (e.g. 0.0, 0.15, 0.30) the same way as the prescaling triage, holding
  the backbone and target fixed. Keep it fixed across the main sweep until then; varying it
  per flavor would confound the report card.
- [ ] Frozen warmup then coadaptation: train for N epochs with `mpnn_lr=0` so the FFN head
  finds a reasonable operating point against the fixed representations, then unfreeze the
  MPNN and continue training at a reduced rate. Avoids the large gradient shock that occurs
  when a randomly initialized head immediately backpropagates into a pretrained backbone,
  while still allowing the MPNN and FFN to coadapt once the head has stabilized. Requires
  a two-phase training schedule not currently supported by the anvil config; likely needs a
  custom Lightning callback or a sequential two-recipe approach.

## Methodology watch-items

- Binary fingerprint targets (ecfp, atompair, pubchem, e3fp) are deterministic from the
  input graph, so they are a leaky and likely weak pretext. Their report-card position is
  a result, not a bug; report it as such.
- 3D flavors depend on generated conformers, so their targets are not bit-reproducible.
  The conformer settings and seed live in `sarizard/pretraining/config.py` (ETKDG + MMFF94,
  `CONFORMER_SEED`); treat reproducibility as approximate.
- Keep the pretraining regime fixed across flavors. The only intended difference is the
  target block and the MSE/BCE choice; any other change confounds the report card.
- Prescaling is part of the fixed regime. Pick one recipe in the milestone-4 triage, bake it
  in (milestone 5), and apply it identically to every continuous flavor; changing prescaling
  mid-sweep confounds the report card the same way changing the backbone would. The triage
  itself varies prescaling only because the backbone, corpus, and target (osmordred) are held
  fixed there.
- The masked-pretext target dropout (`losses.py`, `DROPOUT_FRACTION=0.30`) is a fixed
  fraction applied to every flavor, so its effect scales with target width: near-uniform 70%
  supervision at high dims, noisy and high-variance at low dims (jazzy 6, ml_qm 24). It is
  part of the fixed regime; do not special-case small flavors mid-sweep. See the dropout
  ablation in Future experiments.
