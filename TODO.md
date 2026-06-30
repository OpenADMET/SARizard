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
- [ ] 4. Fan out the direct-compute flavors on the cluster: rdkit2d, erg, ecfp, atompair,
  pubchem, the 3D set (usrcat, whim, e3fp), and jazzy (isolated env for its RDKit pin).
- [ ] 5. Add the learned-model flavors: minimol, surrogate_adme, ml_qm. Each runs its
  source model over the shared corpus in an isolated environment and caches the target.
- [ ] 6. Report card: heatmap of endpoints by flavors with a selectable metric (default R-squared).
- [ ] 7. Meta-model: stack per-flavor finetuned predictions per endpoint, fit LGBM/RF/MLP
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

## Methodology watch-items

- Binary fingerprint targets (ecfp, atompair, pubchem, e3fp) are deterministic from the
  input graph, so they are a leaky and likely weak pretext. Their report-card position is
  a result, not a bug; report it as such.
- 3D flavors depend on generated conformers, so their targets are not bit-reproducible.
  The conformer settings and seed live in `pretraining/config.py` (ETKDG + MMFF94,
  `CONFORMER_SEED`); treat reproducibility as approximate.
- Keep the pretraining regime fixed across flavors. The only intended difference is the
  target block and the MSE/BCE choice; any other change confounds the report card.
