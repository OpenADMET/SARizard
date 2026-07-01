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
  **Redo in progress (250K numbers invalid):** all 7 first-pass pretraining runs diverged
  mid-training (unclipped gradients + a too-dense masked-pretext keep fraction); regime
  reconciled against `../foundation-models/pretraining` and fixed (`config.py`:
  `GRADIENT_CLIP_VAL=0.5`, `PATIENCE=50`, `FNN_HIDDEN_SIZE=1024`, `WARMUP_EPOCHS=2`,
  `DROPOUT_FRACTION=0.85`, bf16/16-mixed precision). Triage is being rerun on the full corpus
  (`corpus/corpus_full.parquet`, 944296 molecules, generated via `CORPUS_FILE`/`CORPUS_N` in
  `slurm/env.sh`) instead of the 250K screening subset, scoped to this triage only, not a
  change to milestones 6-9's corpus plan. `chemeleon_baseline` runs alone first to confirm
  stability before the other six recipes fire. Original 250K artifacts archived at
  `archive/ablation_250k_pre_gradclip/`; see `FINDINGS.md` and `wiki/Prescaling Ablation.md`
  for the full account.
  **Status:** corpus built (job 18097840). The first full-corpus target job (18097911) failed
  on a torn `sarizard-osmordred` env (conda-meta claimed `libparquet` installed but its `.so`
  files were missing and the env was unregistered with `conda env list`, from an interrupted
  prior `setup.sh` run); rebuilt clean with `FORCE=1 bash setup.sh osmordred`. Osmordred target
  computation resubmitted and completed (job 18106135). `chemeleon_baseline`'s first
  prescale/pretrain attempt (jobs 18106233, 18106234) was submitted without `CORPUS_FILE` set,
  so `split.py` paired the full-corpus target against the 250K corpus's SMILES (row-count
  mismatch caught before training started, no bad checkpoint produced); stale split deleted and
  resubmitted with `CORPUS_FILE=corpus/corpus_full.parquet CORPUS_N=1000000` exported (jobs
  18108226, 18108227). Prescale (18108226) succeeded; pretrain (18108227) then failed on a
  second, pre-existing gap: `sarizard` was missing `tensorboard` despite `envs/main.yml`
  declaring it. Fixed with `conda env update -n sarizard -f envs/main.yml`; pretrain resubmitted
  as job 18108864, and started training, but on CPU: a third, pre-existing gap, `sarizard`'s
  PyTorch was the CPU-only conda-forge build (`envs/main.yml` intentionally leaves PyTorch's
  CUDA build to a manual per-cluster install, which was never done). Cancelled 18108864,
  installed `pytorch=2.12.0=cuda129_mkl_py311*` matching the node driver, confirmed
  `torch.cuda.is_available()`, and resubmitted (job 18109452, confirmed running on an A100 GPU).
  `chemeleon_baseline` ran clean through 15 epochs (`val/r2` 0.773 to 0.952, monotonic loss
  decrease, well past the epoch 4-10 collapse window from the 250K runs), confirming the regime
  fix. The other six recipes are now submitted: prescale job 18111455, pretrain job 18111456
  (dependent), covering `minimal`, `order_fix`, `plus_drop_corr`, `plus_drop_low_var`,
  `plus_yeo_johnson`, `full`.
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
- [ ] 10. (GATED on 8, and only if any flavor beats baseline) Scale the flavors that show
  utility up to the full 1M-molecule corpus to produce the final foundation-model artifacts.
  The 250K corpus is the screening set that decides which descriptor targets are worth the
  cost; the sweep is descriptive, not the deliverable. If no flavor clears the baseline on
  250K, there is nothing to scale and this milestone is void. Otherwise, recompute the winning
  flavors' targets over the 1M set, pretrain at full scale, and ship those foundations as the
  release artifacts. Hold the pretraining regime (backbone, prescaling from milestone 5,
  target-dropout) identical to the sweep so the 1M foundation is the same experiment at scale,
  not a new one.

## Open items (need input or external data)

- [x] ML-QM flavor: use qmdesc (Guan et al., MIT, bundled weights). It predicts 4 atom-level
  (partial charge, nucleophilic/electrophilic Fukui, NMR shielding) and 2 bond-level (bond
  order, bond length) QM descriptors. Pooled per descriptor with mean/std/min/max to a
  24-dim molecule target. Open design choice: a richer alternative is to regress the
  per-atom/per-bond descriptors directly as node/edge targets (chemprop supports this),
  which keeps the resolution that pooling discards; revisit if the pooled target underperforms.
- [x] SLURM specifics: partitions are set in the sbatch headers (cpu for corpus/target/split/
  prescale, gpu with `--gres=gpu:1` for pretrain/finetune/analyze), per-job time limits live in
  each header, `setup.sh` builds the conda envs on the cluster, and the repo sits on the shared
  filesystem. No `--account` is required on this cluster; add one to the headers if yours needs it.
- [x] Surrogate-ADME data: download the Novartis Nat Commun 2024 released CSV
  (DOI 10.1038/s41467-024-49979-3, Supplementary Data 1, CC BY 4.0) to
  `cache/surrogate/protacdb2.0_zinc_chembl_dataset.csv` (the default `SURROGATE_CSV`). The CSV is
  the pretraining corpus for this flavor; no model training step is needed.
- [ ] MLIP conformer backend for the 3D flavors (usrcat, whim, e3fp): the calculators
  currently use RDKit ETKDG + MMFF94 (seeded, in `features/skfp_targets.py`). Once the
  pipeline runs end to end, add an ML-potential backend (candidates: Auto3D with
  ANI2x/AIMNet2, or MACE-OFF23 via ASE/openmm-ml) in an isolated GPU env and make it the
  pluggable conformer source; compare descriptor stability against MMFF94. Until then, the 3D
  flavors and jazzy are the slowest targets; shard them across array tasks
  (`slurm/compute_target_shard.sbatch` + `merge_target.sbatch`) to fit the wall clock.

## Future experiments

- [x] Reduced MPNN LR: repeat the full finetuning sweep with `mpnn_lr` set to a fraction of
  `ffn_lr` (1e-4 vs 1e-3) rather than 0. Tests whether partial unfreezing recovers
  performance on endpoints where the frozen backbone underperforms random init, or whether
  it simply reintroduces the initialization-washing problem. Scripted: `bash
  slurm/run_lr_experiments.sh` (mode `reduced`), reusing the flavor foundations; compare in
  `plots/lr_ranking_r2.csv`.
- [x] Fully unlocked MPNN: repeat with `mpnn_lr` equal to `ffn_lr` (1e-3). Establishes the
  upper bound on what full finetuning can achieve and quantifies how much signal the frozen
  protocol sacrifices; the gap between frozen and unlocked is the cost of the clean ablation.
  Scripted alongside reduced (mode `unlocked`) in `slurm/run_lr_experiments.sh`.
- [ ] Prescaling ablation MPNN LR sweep (in progress): cross the milestone-4 prescaling triage with the
  finetune LR modes. The triage originally finetuned every prescaling recipe frozen only
  (`mpnn_lr=0`); this repeats it at `reduced` (`mpnn_lr=1e-4`) and `unlocked` (`mpnn_lr=1e-3`)
  so the preprocessing decision is judged under all three protocols rather than assuming the
  frozen ranking holds once the backbone can move. `sarizard/configs/generate.py` now threads
  `--mpnn-lr-mode` through ablation mode; recipes land in
  `configs/ablation_<name>__s42__{reduced,unlocked}/` and finetune via `ablation_finetune.sbatch`.
  All 504 finetune runs are complete (7 recipes x 3 protocols x 24 endpoints, all result dirs
  present); the remaining step is `ablation_analyze.sbatch`, which collects
  `results/ablation_metrics.csv` and drives the protocol-aware `prescaling_report`. That report
  emits a report card and ranking per protocol plus `plots/prescaling_mode_comparison_<metric>.csv`
  (each recipe's mean metric under frozen, reduced, and unlocked) so the ranking's stability is
  read directly. Analyze is not yet submitted, so those artifacts do not exist yet.
- [x] Multi-seed foundations: pretrain each flavor at several seeds to separate the foundation
  effect from initialization variance. Set `FLAVOR_SEEDS` for `run_all.sh` (and
  `ABLATION_SEEDS` for the triage); the report card and meta-model average the seeds per
  flavor, and re-running with more seeds fills in only the new ones.
- [ ] **Blocker for Milestone 6, raised in urgency:** target-dropout fraction for small
  flavors. The masked-pretext dropout in `losses.py` (`DROPOUT_FRACTION`, applied per target
  element to every flavor) keeps a fixed fraction, not a fixed count. Its rationale (stop the
  head co-adapting across a wide descriptor block) is strong at 3585 dims (osmordred) but
  breaks down at low dims. Since the training-collapse regime fix, `DROPOUT_FRACTION=0.85`
  (keeps 15%, matching `../foundation-models/pretraining`'s `MASKING_RATIO`), so jazzy (6
  dims) now keeps under 1 target/step on average, worse than the previous 0.30 (~4 of 6
  kept) that this item originally flagged as merely noisy. Mechanically safe (loss aggregates
  over all kept elements in the batch, not per-row, so no divide-by-zero), but likely
  unusably sparse. Ablate the fraction (e.g. 0.0, 0.15, 0.85) per small flavor, holding the
  backbone and target fixed, the same way as the prescaling triage, **before** Milestone 6
  fans out jazzy/ml_qm/surrogate_adme, not only "if they underperform" as originally scoped.
  Keep it fixed across the main sweep once decided; varying it per flavor mid-sweep would
  confound the report card.
- [ ] Frozen warmup then coadaptation: train for N epochs with `mpnn_lr=0` so the FFN head
  finds a reasonable operating point against the fixed representations, then unfreeze the
  MPNN and continue training at a reduced rate. Avoids the large gradient shock that occurs
  when a randomly initialized head immediately backpropagates into a pretrained backbone,
  while still allowing the MPNN and FFN to coadapt once the head has stabilized. Requires
  a two-phase training schedule not currently supported by the anvil config; likely needs a
  custom Lightning callback or a sequential two-recipe approach. This is the one LR experiment
  `run_lr_experiments.sh` does not cover (it only sweeps single-rate protocols: reduced,
  unlocked); wiring it needs that anvil feature first.
- [ ] PCA-compressed osmordred target (not yet started): osmordred only, backbone/corpus/regime
  held fixed as usual. Run the descriptor matrix through the full prescaling pipeline (the
  `full` recipe: order-fixed winsorize/z-score plus correlation drop, low-variance drop, and
  Yeo-Johnson), then fit PCA on the resulting matrix and pretrain against the component scores
  instead of the prescaled descriptors themselves. Three iterations, one per explained-variance
  threshold: 80%, 90%, and 95% (each threshold picks its own component count). Tests whether a
  smaller, decorrelated target trains a better, or just cheaper, foundation than the full
  3585-dim block. Also bears on the target-dropout blocker above: a PCA target has far fewer,
  already-decorrelated dimensions, which changes the keep-count-per-step math that motivates
  that ablation. Needs new wiring: an `osmordred_pca<threshold>` target variant computed after
  prescaling (fit PCA on the train split only, apply the same transform to val, cache the
  component count each threshold picks), plus the corresponding prescale/pretrain/finetune/
  analyze plumbing. Plan only; do not execute yet.

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
- The masked-pretext target dropout (`losses.py`, `DROPOUT_FRACTION=0.85` as of the
  training-collapse regime fix, keeps 15%) is a fixed fraction applied to every flavor, so
  its effect scales with target width: reasonable supervision density at high dims
  (osmordred, 3585), likely unusably sparse at low dims (jazzy 6, ml_qm 24, under 1
  target/step on average). It is part of the fixed regime; do not special-case small flavors
  mid-sweep, but see the now-urgent dropout ablation in Future experiments, which must land
  before Milestone 6.
- Pretraining regime constants (`sarizard/pretraining/config.py`) are reconciled against the
  sibling `../foundation-models/pretraining` implementation as of the training-collapse fix
  (`PATIENCE`, `FNN_HIDDEN_SIZE`, `WARMUP_EPOCHS`, `DROPOUT_FRACTION`, `GRADIENT_CLIP_VAL`,
  precision). Treat that sibling as the reference for any future regime question; a
  divergence from it is now a deliberate choice, not an oversight, and should be commented
  as such.
