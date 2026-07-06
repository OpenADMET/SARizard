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
- [x] 4. Prescaling ablation triage (runs BEFORE the flavor sweep). Drive osmordred through
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
  All 7 pretrain runs (18109452, 18111456 array 0/2-6) completed clean overnight: full 100
  epochs each, stable losses, no collapse; all 7 `foundations/ablation_<name>__s42_mp.pt`
  checkpoints on disk. Generated the 168 finetune recipes (7 recipes x 24 endpoints,
  `configs/ablation_<name>__s42/`) and submitted the finetune array (job 18410392,
  `--array=0-167`). All 168 tasks failed identically on an unrelated `openadmet` env gap
  (mismatched `boto3`/`botocore` breaking `openadmet anvil`'s CLI import); fixed with
  `pip install --upgrade --force-reinstall boto3` in the `openadmet` env, verified with a
  direct run, then resubmitted (finetune job 18420137, analyze job 18420145 chained
  `afterok`). Both completed clean. **Triage complete: `chemeleon_baseline` wins** by mean
  R-squared under the frozen protocol (0.352 vs. 0.307-0.342 for the other six recipes); see
  `FINDINGS.md` for the full ranking and `results/ablation_metrics.csv` /
  `plots/prescaling_report_r2.csv` for the numbers.
- [x] 5. (GATED on 4) Harden the chosen prescaling into the core flavor-sweep workflow. Wire
  the winning `PrescalingConfig` into the default `split.py` path (or insert a prescale step
  ahead of it) so every flavor pretrains on the same, vetted preprocessing. Until this lands,
  the flavor sweep keeps the current `chemeleon_baseline` behavior. Record the decision and
  the margin over baseline in `FINDINGS.md`.
  **Decision: `chemeleon_baseline`, no code change needed.** `split.py` already reproduces
  `chemeleon_baseline` (mean/std on the raw target reused for both winsorization and
  z-scoring), and that recipe won the triage, so the default path is already the vetted one.
  The flavor sweep (Milestone 6) can proceed unmodified. The subsequent cross-protocol LR
  sweep (Future experiments, below) confirmed the frozen-protocol win but found the margin
  over `order_fix` narrow, not decisive; see `FINDINGS.md` for the full picture.
  **250K corpus-size check (in progress):** repeats the same 7 recipes x 3 protocols triage
  on the original 250K screening corpus (`corpus/corpus_250k.parquet`, the milestone-2
  default) instead of the full corpus, now that the regime fix makes the run valid there too
  (the original 250K triage predates the fix and is archived, invalid). Tests whether the
  full-corpus ranking (`chemeleon_baseline` narrowly over `order_fix`) holds at 1/4 the
  corpus size, or whether corpus size itself was doing some of the work. Before submitting,
  archived the full-corpus run's artifacts to `archive/ablation_full_corpus/` (targets,
  `cache/ablations/`, foundations, configs, results, plots, tensorboard runs), mirroring the
  existing `archive/ablation_250k_pre_gradclip/` precedent, so the 250K rerun does not
  overwrite the full-corpus foundations `FINDINGS.md` already cites. Submitted in one chain
  via `bash slurm/run_ablations.sh` with `ABLATION_LR_MODES="frozen reduced unlocked"` set
  upfront (corpus job 18455150, target 18455151, prescale 18455152, pretrain 18455153,
  finetune 18455154 array 0-503, analyze 18455155, each `afterok` the last). Hit one
  submission-time snag: running `run_ablations.sh` interactively inherited a stale
  `SLURM_SUBMIT_DIR` from the enclosing interactive SLURM allocation (`/home/westd1`, not the
  repo), which `slurm/env.sh` prefers over `pwd` for `REPO_DIR`; fixed by exporting
  `REPO_DIR=/scratch/choderaj/westd/SARizard` explicitly before invoking the script.
  **Result: the 250K ranking does not match the full-corpus ranking.** All 504 finetunes and
  the chained analyze completed clean (no failures). Mean R-squared per recipe per protocol
  (`plots/prescaling_mode_comparison_r2.csv`): frozen winner is `plus_drop_low_var` (0.347,
  vs. 0.352 for `chemeleon_baseline` on the full corpus), reduced winner is
  `plus_yeo_johnson` (0.381), unlocked winner is `minimal` (0.324). None of the three match
  their full-corpus counterparts (`chemeleon_baseline`, `order_fix`, `plus_drop_corr`); most
  strikingly, `chemeleon_baseline` (the full-corpus frozen winner and the current Milestone-5
  decision) drops to 5th of 7 under frozen at 250K (0.295). Corpus size is doing real work in
  which prescaling recipe wins, not just adding statistical power to the same ranking. Does
  not change the Milestone-5 decision as recorded (the flavor sweep runs on the full corpus,
  so the full-corpus frozen ranking is the relevant one), but it means the ranking should not
  be assumed to generalize to a different corpus size, e.g. if the full 1M corpus is ever
  substituted in milestone 10. See `FINDINGS.md` for the full tables.
- [x] 6. Fan out the direct-compute flavors on the cluster: rdkit2d, erg, ecfp, atompair,
  pubchem, the 3D set (usrcat, whim, e3fp), and jazzy (isolated env for its RDKit pin).
  Fired without the target-dropout-fraction ablation below (explicit call:
  ship milestone 6 now, revisit jazzy's dropout fraction later if it underperforms rather than
  gating on it). All 9 flavors use `order_fix` prescaling (not `chemeleon_baseline`) for their
  continuous targets, on explicit instruction, overriding the Milestone-5 recipe for this
  sweep; binary/fingerprint flavors (ecfp, atompair, pubchem, e3fp) are unaffected (they skip
  rescaling regardless). All three finetune protocols (frozen, reduced, unlocked) run from the
  start rather than frozen-only-then-follow-up.
  All 9 flavors' targets were already cached on the 250K corpus from earlier scaffolding
  work, so `compute_targets` skipped straight through. Two small code changes were needed to
  fire this off: (1) `slurm/env.sh`'s `flavor_list()` gained an optional `FLAVOR_SUBSET`
  filter (space-separated flavor names; unset keeps the full registry) so the array-sized
  stages (targets/split/pretrain/finetune/analyze) can be scoped to just these 9 flavors
  without dragging in `osmordred` (already done) or the milestone-7 model flavors
  (`minimol`/`surrogate_adme`/`ml_qm`, not yet started); scoping matters because `analyze`'s
  `afterok` dependency needs every array task to succeed, and running unbuilt milestone-7
  flavors alongside would risk cancelling the whole chain on an unrelated failure.
  (2) `slurm/split.sbatch` now runs `prescaling.py --ablation order_fix` ahead of `split.py
  --prescaled` for continuous flavors (checked via `get_flavor(flavor).kind`), instead of
  calling `split.py` directly; binary flavors keep the old direct path unchanged. This is now
  the default prescaling for every future continuous-flavor split through this script, not
  just milestone 6's.
  Submitted as one dependency chain: corpus (job 18564905, skipped, already exists) → targets
  (18564906, array 0-8, skipped, already cached) → split (18564907, array 0-8, confirmed
  `order_fix` running clean on the 5 continuous flavors and binary skip-rescaling on the other
  4) → pretrain (18564908, array 0-8) → finetune (18565234, frozen, array 0-215, 216 recipes)
  and lr-finetune (18565235, reduced+unlocked, array 0-431, 432 recipes) in parallel off the
  same foundations → analyze (18565236, after finetune) and lr-analyze (18565237, after both
  finetune jobs). 648 finetune recipes total (9 flavors x 24 endpoints x 3 protocols).
  **Complete: every stage of the chain finished clean (exit code 0), no failures.**
  `results/metrics.csv` (frozen, 288 rows) and `results/lr_metrics.csv` (reduced +
  unlocked, 864 rows) cover all 9 flavors x 32 endpoint-columns. `rdkit2d` is the
  strongest flavor overall (mean R-squared 0.350 frozen, 0.371 reduced), winning 6 of 8
  endpoint families; `usrcat` is the specialization result, winning potency and hERG
  specifically despite mid-table overall performance. `reduced` is again the best
  protocol on average, and `unlocked` again compresses the spread between flavors,
  matching the prescaling-triage LR-mode pattern. Full report card and per-flavor read in
  `FINDINGS.md`. This flavor sweep ran on the 250K corpus per Milestone 2's scope (not the
  full corpus used for the Milestone-4/5 osmordred triage), so osmordred is not yet
  directly comparable on this table; a controlled comparison needs osmordred rerun under
  this same protocol (250K corpus, `order_fix`).
- [ ] 7. Add the learned-model flavors: minimol, surrogate_adme, ml_qm. Each runs its
  source model over the shared corpus in an isolated environment and caches the target.
  **In progress, scoped to minimol only.** `ml_qm` (24-dim target) and `surrogate_adme`
  (25-dim target) are skipped for now: the target-dropout-fraction blocker below names both
  by name as needing that ablation before fan-out, not only "if they underperform," and it
  has not been run. Explicit call: hold both pending either that ablation or an override
  decision, rather than repeat the jazzy precedent (milestone 6 shipped jazzy, a similarly
  narrow target, without the ablation). `minimol` (512-dim) is not implicated by the blocker
  (comparable width to osmordred's 3585), so it proceeds alone via
  `FLAVOR_SUBSET=minimol bash slurm/run_all.sh` (single frozen protocol, no LR-mode
  override, unlike milestone 6). Submitted as one chain: corpus (job 19181243, skipped,
  already exists) → targets (19181244) → split (19181245) → pretrain (19181246) → finetune
  (19181247, array 0-23, 24 recipes) → analyze (19181248), each `afterok` the last.
  Found and fixed a real bug before submitting: `cache/targets/minimol/target.npy` (250K
  rows, cached from earlier scaffolding) was entirely NaN. `envs/minimol.yml` left `scipy`
  unpinned, so pip resolved 1.15.3, which dropped `float16` sparse-matrix support that
  `graphium`'s featurizer (a minimol dependency) relies on; every calculator call raised
  `ValueError` before ever reaching a molecule, and the whole target came back NaN with no
  per-row signal to catch it (this predates `compute_target.py`'s later crash-orphan guard,
  or was produced before that guard existed). Pinned `scipy<1.13` in `envs/minimol.yml`,
  confirmed the calculator returns real embeddings, deleted the corrupted
  `target.npy`/`target.zarr`, and let the resubmitted targets job recompute it clean.
  **Chain complete through finetune; analyze's "failure" is expected.** Corpus, targets,
  split, pretrain, and all 24 finetune array tasks (19181247) completed clean. Analyze
  (19181248) shows FAILED in `sacct`, but the report card stage ran first and wrote
  `results/metrics.csv` (32 rows) and the report-card plots before the job exited nonzero;
  the actual exit came from `meta_model.py:210`'s deliberate
  `raise SystemExit("no stackable endpoints... need >=2 flavors each")`, since minimol is
  the only flavor with results in that directory right now. That guard is doing its job,
  not signaling a bug. minimol's mean R-squared across the 32 endpoint-columns is 0.360
  (frozen protocol), edging out `rdkit2d`'s 0.350 frozen mean from Milestone 6, though the
  two runs were not evaluated in one merged table so treat this as context, not a
  controlled comparison yet.
  **Note for the next merged evaluation:** `evaluate.py`'s default `--out` always writes
  `results/metrics.csv`, so this run's `--flavors minimol` invocation overwrote Milestone
  6's 288-row merged file with minimol's 32 rows. Nothing is lost: every Milestone-6
  finetune result dir and its cached `y_pred.npy` are still on disk under `results/`, and a
  separately-named `results/metrics_report_card_with_osmordred.csv` already holds a
  320-row merged table. Re-running `evaluate.py` with no `--flavors` filter (or with the
  full flavor list) will regenerate a complete `results/metrics.csv` from the cached
  predictions; do that before the next report-card or meta-model pass so it is not
  scoped to whichever flavor last ran the analyze stage.
- [x] 8. Report card: heatmap of endpoints by flavors with a selectable metric (default R-squared).
  Regenerated across all 10 completed flavors (the 9 Milestone-6 flavors plus `minimol`) by
  resubmitting `analyze.sbatch` with no `FLAVOR_SUBSET` (job 19230968, completed clean);
  `results/metrics.csv` (320 rows) and `plots/report_card_r2.png`/`.csv` are the current
  merged report card. `osmordred`, `ml_qm`, and `surrogate_adme` are still absent (osmordred
  ran under a different corpus/naming convention, the other two haven't fanned out yet).
- [x] 9. Meta-model: stack per-flavor finetuned predictions per endpoint, fit LGBM/RF/MLP
  on out-of-fold predictions, compare to the best single flavor.
  First real result, produced by the same job 19230968 now that ≥2 flavors have results:
  the LGBM meta-model beats the best single flavor on 23 of 32 endpoint-columns, mean
  R-squared 0.481 vs. 0.390 for the best single flavor per endpoint (mean delta +0.091).
  `rdkit2d` (11 endpoints) and `minimol` (9) are the most frequent single-flavor winners the
  meta-model has to beat; `usrcat` (4), `atompair`/`jazzy`/`e3fp` (2 each), `whim`/`ecfp` (1
  each) round out the rest. See `results/meta_model_lgbm.csv` for the per-endpoint table and
  `FINDINGS.md` for the read.
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
- [x] Prescaling ablation MPNN LR sweep: cross the milestone-4 prescaling triage with the
  finetune LR modes. The triage originally finetuned every prescaling recipe frozen only
  (`mpnn_lr=0`); this repeats it at `reduced` (`mpnn_lr=1e-4`) and `unlocked` (`mpnn_lr=1e-3`)
  so the preprocessing decision is judged under all three protocols rather than assuming the
  frozen ranking holds once the backbone can move. `sarizard/configs/generate.py` threads
  `--mpnn-lr-mode` through ablation mode; recipes land in
  `configs/ablation_<name>__s42__{reduced,unlocked}/` and finetune via `ablation_finetune.sbatch`.
  All 504 finetune runs completed (7 recipes x 3 protocols x 24 endpoints, job 18443536, no
  failures), and the chained analyze job (18443537) collected `results/ablation_metrics.csv`
  (672 rows) and wrote `plots/prescaling_mode_comparison_r2.csv`.
  **Result: the winning recipe shifts by protocol** (`chemeleon_baseline` wins frozen,
  `order_fix` wins reduced, `plus_drop_corr` wins unlocked), and `chemeleon_baseline` vs.
  `order_fix` (the two closest recipes) are close enough that a ranked-choice election pooling
  all three protocols came down to 44-43 in `order_fix`'s favor, despite `chemeleon_baseline`
  leading every round but the last. `reduced` is uniformly the best protocol for every recipe;
  `unlocked` compresses the spread between recipes, so prescaling matters less once the
  backbone can fully adapt. `plus_drop_low_var` is the clear bottom performer across every
  protocol and every read. Does not overturn the Milestone-5 decision (`chemeleon_baseline`
  wins frozen, the sweep's protocol) but downgrades it from "clear winner" to "narrow winner
  over `order_fix`"; see `FINDINGS.md` for the full cross-protocol tables.
- [x] Multi-seed foundations: pretrain each flavor at several seeds to separate the foundation
  effect from initialization variance. Set `FLAVOR_SEEDS` for `run_all.sh` (and
  `ABLATION_SEEDS` for the triage); the report card and meta-model average the seeds per
  flavor, and re-running with more seeds fills in only the new ones.
- [ ] **Blocker for Milestone 7, raised in urgency:** target-dropout fraction for small
  flavors. The masked-pretext dropout in `losses.py` (`DROPOUT_FRACTION`, applied per target
  element to every flavor) keeps a fixed fraction, not a fixed count. Its rationale (stop the
  head co-adapting across a wide descriptor block) is strong at 3585 dims (osmordred) but
  breaks down at low dims. Since the training-collapse regime fix, `DROPOUT_FRACTION=0.85`
  (keeps 15%, matching `../foundation-models/pretraining`'s `MASKING_RATIO`), so jazzy (6
  dims) now keeps under 1 target/step on average, worse than the previous 0.30 (~4 of 6
  kept) that this item originally flagged as merely noisy. Mechanically safe (loss aggregates
  over all kept elements in the batch, not per-row, so no divide-by-zero), but likely
  unusably sparse. Ablate the fraction (e.g. 0.0, 0.15, 0.85) per small flavor, holding the
  backbone and target fixed, the same way as the prescaling triage, **before** ml_qm (24
  dims) or surrogate_adme (25 dims) fan out, not only "if they underperform" as originally
  scoped. Keep it fixed across the main sweep once decided; varying it per flavor mid-sweep
  would confound the report card.
  **Status: still open, now the reason ml_qm and surrogate_adme are held out of Milestone 7.**
  Milestone 6 shipped `jazzy` without this ablation (an explicit, recorded deferral for that
  one flavor); Milestone 7 does not repeat that deferral for `ml_qm`/`surrogate_adme` since
  this item names them directly. `minimol` (512 dims) is unaffected by this blocker and
  proceeds on its own.
  **Decision: override to 0.0 for these two flavors, skipping the ablation, not running it
  later.** Explicit call, not a default: the ablation (0.0 vs. 0.15 vs. 0.85) is not being run;
  0.0 is picked directly on the reasoning that under-1-target/step supervision is unlikely to
  beat no masking at all for a 24-25 dim block, and confirmed post hoc rather than compared
  against alternatives. `DROPOUT_FRACTION` was hardcoded and applied identically to every
  flavor with no per-flavor override path, so wired one rather than changing the global
  constant (which would silently change the regime for every already-run flavor):
  `losses.py`'s `_RandomDropoutMixin` now takes a `dropout_fraction` constructor arg
  (default the shared constant; survives `torchmetrics.Metric.clone()`, which is a
  `deepcopy`, so the override is preserved on the cloned validation-metric instance),
  `train.py`'s `_build_model` and `main` thread a new `--dropout-fraction` CLI flag through
  to it and record the value in `foundation.json`, and `slurm/env.sh` gained
  `DROPOUT_FRACTION_OVERRIDES` (space-separated `flavor=value` pairs) plus a
  `dropout_fraction_for` lookup that `pretrain.sbatch` consults per flavor. Every flavor
  without an entry (everything run so far) still pretrains at the regime default, so this
  does not touch anything already on disk. To fan out `ml_qm`/`surrogate_adme` at 0.0,
  export `DROPOUT_FRACTION_OVERRIDES="ml_qm=0.0 surrogate_adme=0.0"` before submitting.
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
