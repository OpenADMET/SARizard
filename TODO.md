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
- [ ] Surrogate-ADME inference (resolved approach, needs data + a training run): the
  Novartis Nat Commun 2024 work (DOI 10.1038/s41467-024-49979-3, CC BY 4.0) ships NO
  runnable model, only a 273,706-row CSV of 25 precomputed ADME predictions
  (Supplementary Data 1) plus a chemprop v1.6.1 recipe. To keep this flavor on the shared
  corpus, `features/surrogate_target.py` retrains a surrogate multitask D-MPNN from that
  CSV and predicts it over our 250K (documented deviation: a single 25-task model rather
  than the paper's four per-group models). Remaining: download the dataset, run
  `surrogate_target.train_surrogate`, then `compute_target --flavor surrogate_adme`.
- [ ] MLIP conformer backend for the 3D flavors (usrcat, whim, e3fp): the calculators
  currently use RDKit ETKDG + MMFF94 (seeded, in `features/skfp_targets.py`). Once the
  pipeline runs end to end, add an ML-potential backend (candidates: Auto3D with
  ANI2x/AIMNet2, or MACE-OFF23 via ASE/openmm-ml) in an isolated GPU env and make it the
  pluggable conformer source; compare descriptor stability against MMFF94.

## Methodology watch-items

- Binary fingerprint targets (ecfp, atompair, pubchem, e3fp) are deterministic from the
  input graph, so they are a leaky and likely weak pretext. Their report-card position is
  a result, not a bug; report it as such.
- 3D flavors depend on generated conformers, so their targets are not bit-reproducible.
  The conformer settings and seed live in `pretraining/config.py` (ETKDG + MMFF94,
  `CONFORMER_SEED`); treat reproducibility as approximate.
- Keep the pretraining regime fixed across flavors. The only intended difference is the
  target block and the MSE/BCE choice; any other change confounds the report card.
