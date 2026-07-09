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
- [x] 2. Prepare the shared 250K corpus from the original CheMeleon PubChem set (Zenodo
  DOI 10.5281/zenodo.15733574), single seed, persisted, with cached default-featurizer graphs.
  **Superseded: the shared corpus is now the full corpus, not the 250K screening set.**
  Decision to run the whole flavor sweep (Milestones 6-9, not just the Milestone 10
  scale-up) directly on `corpus/corpus_full.parquet` (944,296 molecules, `CORPUS_FILE`/
  `CORPUS_N` in `slurm/env.sh`). This folds what Milestone 10 originally deferred
  ("scale winning flavors up to 1M after the 250K screen decides which are worth it")
  forward into the main sweep itself. The 250K flavor-sweep artifacts (targets, splits,
  foundations, results, plots) are archived at `archive/flavor_sweep_250k/` rather than
  overwritten, mirroring the ablation-triage archive precedent. See Milestones 6-9 below
  for the rerun's progress.
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
  (`minimol`/`surrogate_adme`, not yet started); scoping matters because `analyze`'s
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
  **Rerunning on the full corpus (see Milestone 2's supersede note).** 250K artifacts
  archived to `archive/flavor_sweep_250k/`. Targets for all 9 flavors recomputed against
  `corpus/corpus_full.parquet`; the fast direct-compute ones (atompair, ecfp, erg, pubchem,
  rdkit2d) finished within the hour, the sharded slow ones (usrcat, whim, e3fp, jazzy) via
  `compute_target_shard.sbatch`/`merge_target.sbatch` took several hours each.
  **Submitted the rest of the chain** via `bash slurm/run_all.sh` (`CORPUS_FILE=corpus/
  corpus_full.parquet CORPUS_N=1000000`, the full registry as of submission including
  Milestone 7's `minimol`/`surrogate_adme` and the new `osmordred_pca80/90/95`; `ml_qm` was
  in this chain too but has since been dropped, see Milestone 7): corpus (job
  33586, completed, skipped) → targets (33587, array 0-15, completed, all skipped since
  already cached) → pca-targets (33588) → split (33589, array 0-15) → pretrain (33590,
  16 tasks) → finetune (33591, array 0-383, 384 recipes) → analyze (33592), each `afterok`
  the last. Not yet complete as of this note; check `sacct -j
  33586,33587,33588,33589,33590,33591,33592` for current state.
  **This `run_all.sh` chain covers the frozen protocol only** (384 = 16 flavors x 24
  endpoints x 1 seed). Pretrain task 14 (`osmordred_pca90`) failed transiently on a bad GPU
  node (`No CUDA GPUs are available`), which cascaded to cancel finetune (33591) and analyze
  (33592) via their `afterok` dependency; the failed pretrain was resubmitted as job 93729_14.
  **93729_14 completed clean** (2026-07-07T15:39, 3h41m on an A100), writing
  `foundations/osmordred_pca90__s42_mp.pt`. **All 15 in-scope full-corpus foundations are on
  disk** (osmordred, the 9 direct-compute flavors, minimol, surrogate_adme, and
  osmordred_pca80/90/95; `ml_qm`'s foundation was also written but came back all-NaN and the
  flavor is dropped, see Milestone 7), so the pretrain stage of the full-corpus rerun is complete and the
  finetune stages are unblocked. Still to submit (nothing is queued): the frozen finetune +
  analyze (the cancelled 33591/33592 equivalents), AND the reduced + unlocked protocols
  separately via `bash slurm/run_lr_experiments.sh` (`LR_MODES="reduced unlocked"`,
  `FLAVOR_SEEDS` matching the sweep) off the same foundations, so all three learning-rate
  protocols are covered on the full corpus (per the standing directive in Methodology
  watch-items). Neither is submitted yet.
  **Frozen finetune + analyze now complete (2026-07-08).** The frozen finetune array ran as
  job 477538: 1104 tasks clean, the only failures being `ml_qm`'s 24-endpoint block (tasks
  288-311), expected since that flavor is dropped and its foundation is all-NaN. Analyze was
  submitted as job 509950 but died on a bad GPU node (`iscn008`, `CUDA driver initialization
  failed` on every `model.predict`, after Lightning reported the GPU as available); it wrote a
  degenerate `results/metrics.csv` (1 flavor, from osmordred's stale cached predictions) before
  exiting. Resubmitted with `--exclude=iscn008` as job 510881, which completed clean on
  `iscn010`: `results/metrics.csv` (512 rows = 15 live flavors + `chemeleon_stock`, `ml_qm`
  absent), `plots/report_card_r2.png`/`.csv`, and `results/meta_model_lgbm.csv`. Frozen mean
  R-squared per flavor: surrogate_adme 0.369, minimol 0.355, rdkit2d 0.339, osmordred 0.327,
  osmordred_pca95/90/80 0.325/0.322/0.321, chemeleon_stock 0.297, jazzy 0.284, usrcat 0.281,
  erg 0.271, pubchem 0.270, atompair 0.261, e3fp 0.239, ecfp 0.223, whim 0.204. The LGBM
  meta-model beats the best single flavor on 24 of 32 endpoint-columns (mean delta R-squared
  +0.118). Still to submit: the reduced + unlocked protocols via `slurm/run_lr_experiments.sh`
  (`LR_MODES="reduced unlocked"`), per the all-three-protocols directive.
  **Reduced protocol submitted (2026-07-09).** Sent reduced only (unlocked deferred, held for a
  later submission on explicit call) via `REPO_DIR=/scratch/choderaj/westd/SARizard
  LR_MODES="reduced" FLAVOR_SEEDS="42" bash slurm/run_lr_experiments.sh`: lr-finetune job 605427
  (array 0-359 = 15 flavors x 24 endpoints x 1 seed), lr-analyze job 605428 chained `afterok`.
  Cleared the stale 250K LR recipe dirs first (`configs/lr_reduced__*`, `configs/lr_unlocked__*`,
  which still included the dropped `ml_qm` and, being globbed by `lr_recipe_list`, would have
  swept unlocked too and cascade-cancelled analyze on the all-NaN `ml_qm` foundation); the
  regenerated reduced recipes are the 15-flavor registry and point at the full-corpus
  `foundations/<flavor>__s42_mp.pt`. Unlocked still to submit the same way when reduced looks
  healthy; the per-protocol stock baseline / meta-model and the six report cards (Milestone 8)
  remain gated behind both non-frozen protocols.
  **Reduced finetune hit the bad-node fault; reran the 10 casualties (2026-07-09).** 605427
  finished 350/360; the 10 failures (minimol, osmordred, osmordred_pca90/95, surrogate_adme
  endpoints) all died on `iscn008` with `CUDA driver initialization failed`, the same bad GPU
  node that killed the frozen analyze (509950), not a code fault. The failures tripped
  lr-analyze 605428's `afterok`, leaving it `DependencyNeverSatisfied`. Each crashed task had
  left a partial `results/lr_reduced__<flavor>__s42/<endpoint>/` dir (dataloaders only, no
  `model.pth`) that the `[[ -d "$OUT" ]]` skip guard in `lr_finetune.sbatch` would have silently
  no-op'd, so removed all 10 first. Cancelled 605428, resubmitted the 10 indices
  (`--array=130,131,133,189,190,192,239,292,293,295 --exclude=iscn008`) as lr-finetune job
  609466, re-chained lr-analyze job 609467 (`afterok`, also `--exclude=iscn008`).
  **Reduced protocol complete (2026-07-09).** 609466 (10 tasks) and 609467 both completed clean.
  `results/lr_metrics.csv` now carries all 15 `lr_reduced__<flavor>__s42` flavors at a uniform 32
  endpoint-columns each, zero NaN (the 10 rerun endpoints all filled); the analyze also
  re-evaluated the 15 frozen flavor dirs, so the file is 960 rows (30 flavor labels x 32).
  `plots/lr_ranking_r2.csv`: reduced mean R-squared 0.324 vs. frozen 0.289 (+0.035, reduced wins
  202 of 285 endpoint comparisons), matching the standing "reduced is the best protocol" pattern.
  Reduced per-flavor leaders: minimol 0.380, surrogate_adme 0.377, rdkit2d 0.371, osmordred 0.360.
  Unlocked still to submit the same way; the Milestone-8 per-protocol stock baseline / meta-model
  and six report cards remain gated behind unlocked.
- [ ] 7. Add the learned-model flavors: minimol, surrogate_adme. Each runs its
  source model over the shared corpus in an isolated environment and caches the target.
  **`ml_qm` dropped from scope (decision, 2026-07-08): we are not running it.** Its qmdesc
  target legitimately contains ~1.4% NaN (qmdesc fails on some molecules), which, combined with
  the pre-fix prescaling re-scatter bug, reached the trainer and drove `train_loss=nan` from
  epoch 0, producing an all-NaN foundation; rather than rerun it, the flavor is removed from
  the registry, the code (`flavors.py`, `compute_target.py`, `qmdesc_target.py`), and the
  environments (`envs/mlqm.yml`). The sweep is now 15 flavors. The rest of this milestone
  concerns `minimol` and `surrogate_adme` only.
  **In progress, scoped to minimol only.** `surrogate_adme`
  (25-dim target) was skipped in the 250K pass: the target-dropout-fraction blocker below named
  it as needing that ablation before fan-out, not only "if they underperform," and it
  had not been run. Explicit call at the time: hold it pending either that ablation or an override
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
  **Superseded by the full-corpus rerun (Milestone 2).** minimol's 250K results are
  archived at `archive/flavor_sweep_250k/`; its target has been recomputed against
  `corpus/corpus_full.parquet`. `surrogate_adme` is no longer held out: the
  target-dropout-fraction blocker below is resolved by an automatic rule
  (`DROPOUT_OVERRIDE_MAX_DIM=30` in `config.py`, `train.py` falls back to
  `dropout_fraction=0.0` for any target at or under that width) rather than the earlier
  per-flavor `DROPOUT_FRACTION_OVERRIDES` list, so it proceeds alongside every other
  flavor in the full-corpus rerun. `surrogate_adme` keeps its own
  Novartis-molecule corpus (unaffected by the full-corpus switch, per the AGENTS.md
  invariant) and has not been touched. Prescale/split/pretrain/finetune/analyze for
  `surrogate_adme` (and every other flavor) are submitted as part of the same chain described in
  Milestone 6's note above (jobs 33586-33592). **Pretrain is now complete:
  `minimol` and `surrogate_adme` foundations are on disk (part of the 15 in-scope full-corpus
  foundations).** Finetune + analyze still to be submitted (see Milestone 6's note).
- [x] 8. Report card: heatmap of endpoints by flavors with a selectable metric (default R-squared).
  Regenerated across all 10 completed flavors (the 9 Milestone-6 flavors plus `minimol`) by
  resubmitting `analyze.sbatch` with no `FLAVOR_SUBSET` (job 19230968, completed clean);
  `results/metrics.csv` (320 rows) and `plots/report_card_r2.png`/`.csv` are the current
  merged report card. `osmordred` and `surrogate_adme` are still absent (osmordred
  ran under a different corpus/naming convention, surrogate_adme hasn't fanned out yet).
  **Added two reference columns, code done, data not yet generated.** The report card now
  appends the stock-CheMeleon baseline and the LGBM meta-model as labeled columns after a
  blank spacer (divider lines, bold labels), separate from the per-flavor columns since
  neither is a selectable flavor (the baseline used a different corpus/regime; the
  meta-model needs every flavor's prediction, not one deployable foundation). Wiring:
  `report_card.py`'s `build_reference_series`/`meta_model_series`/`augment_with_references`;
  `configs/generate.py --stock-baseline`; `slurm/run_stock_baseline.sh` +
  `finetune_stock_baseline.sbatch` (one-time, corpus/regime-independent); `analyze.sbatch`
  folds `results/chemeleon_stock/` in automatically when present.
  **Stock-baseline finetune submitted; a dedicated 250K report card is chained after it.**
  `bash slurm/run_stock_baseline.sh` submitted job 53237 (24-task array, `results/
  chemeleon_stock/`). `slurm/report_card_250k.sbatch` (job 54188, `afterok:53237`) evaluates
  the baseline into `results/metrics_chemeleon_stock.csv`, merges it with the archived 250K
  flavor-sweep metrics (`archive/flavor_sweep_250k/results/metrics.csv`, 10 flavors) into a
  new derived file (`archive/flavor_sweep_250k/results/metrics_with_references.csv`, the
  archive's own files untouched), and renders `plots/report_card_250k_r2.png` with both
  reference columns, using the archived `meta_model_lgbm.csv` for the meta-model column.
  This is a one-off request (a report card for the archived 250K run specifically), separate
  from the full-corpus rerun's own `analyze` stage (job 33592), which will pick up
  `chemeleon_stock` into the live `results/metrics.csv` automatically once both finish.
  **One stock-finetune task failed transiently; rerun and report card resubmitted.** Task
  `53237_11` (`chembl_clint_rlm_st`) died on `RuntimeError: No CUDA GPUs are available` (a bad
  GPU node, not a code fault); the other 23 tasks completed. The `afterok:53237` dependency
  therefore resolved to `(failed)` and left the report card (job 54188) parked in PENDING
  forever, so nothing was rendered. The crash left a partial `results/chemeleon_stock/
  chembl_clint_rlm_st/` (9 data-prep files, no trained model), which would have made the
  skip-if-exists guard silently no-op a rerun; removed it, then resubmitted just that endpoint
  (`sbatch --array=11`, job 137151_11, completed clean in 1:21). Cancelled the dead 54188 and
  resubmitted the report card chained to the rerun (job 137211, `afterok:137151`).
  **137211 exposed a real render bug, now fixed.** The rerun completed and 137211's evaluate
  and merge stages wrote `results/metrics_chemeleon_stock.csv` (32 rows) and
  `archive/flavor_sweep_250k/results/metrics_with_references.csv` (352 rows, 11 flavors), but
  the render step crashed in `report_card.py`'s `augment_with_references` with
  `ValueError: cannot reindex on an axis with duplicate labels`. Cause: the report card's row
  identity is `dataset · endpoint`, but that pair is not unique when the same endpoint appears
  under both a single-task and a multi-task recipe (`LOG_CLint_HLM`, `LOG_CLint_RLM`, `LogD`
  each collide). `build_matrix` already collapses these with `aggfunc="mean"`, but
  `build_reference_series` and `meta_model_series` built their Series with the raw, duplicated
  index, so `.reindex` onto the pivot failed. This was the first run to exercise the reference
  columns with real data (Milestone 8 left them "code done, data not generated"). Fixed both
  builders to `.groupby(level=0).mean()` the duplicate labels, matching `build_matrix`.
  Verified by rendering locally against 137211's own merged CSV: 29 unique endpoint rows, 13
  columns (10 flavors + spacer + 2 reference columns, both 29/29 populated). Report card saved
  at `plots/report_card_250k_r2.png`/`.csv`; the two intermediate CSVs above are on disk.
  **Final full-corpus report-card deliverable (requested, produce after the sweep analysis
  completes).** Six report cards: one per learning-rate protocol (frozen, reduced, unlocked)
  crossed with the two color modes, `--color-mode absolute` (fixed [0,1] scale) and
  `--color-mode baseline-diverging` (red/blue centered on each row's stock-CheMeleon
  baseline). The stock baseline is finetuned under each protocol (decision), so a card
  diverges around its own-protocol stock baseline, not a single shared frozen reference; the
  meta-model and stock-baseline reference columns on each card are that protocol's.
  Every card shows all 15 flavors (osmordred, the 9 direct-compute flavors,
  minimol, surrogate_adme, and osmordred_pca80/90/95) plus the two reference columns
  (stock baseline, meta-model). Both color modes already exist in `report_card.py`; three gaps
  remain for the non-frozen cards, the per-protocol meta-model, and the per-protocol stock
  baseline, not yet wired:
  (1) `report_card.py` keys columns by flavor label and reads a single `--metrics-csv`, so the
  frozen card runs off `results/metrics.csv` directly, but the reduced and unlocked cards need
  `results/lr_metrics.csv` filtered to one mode with its `lr_<mode>__` prefix stripped back to
  the bare flavor name so the columns match the registry; no `--lr-mode` filter flag exists
  yet, so add one (or a small per-mode CSV pre-filter). (2) The meta-model must run once per
  protocol (see Milestone 9); `meta_model.py` currently runs frozen only and always writes
  `<results>/meta_model_<estimator>.csv`, so each protocol needs its own mode-scoped prediction
  dirs (`--results`/`--flavors`) and a distinct output CSV so the three do not overwrite each
  other, then each card's `--meta-model-csv` points at the matching protocol's file.
  (3) The stock baseline must be finetuned once per protocol. `generate.py --stock-baseline`
  already accepts `--mpnn-lr-mode`, but stock-baseline mode hardcodes the label to
  `chemeleon_stock` and ignores `--label-prefix`, so reduced/unlocked would only change
  `mpnn_lr` inside the recipe while overwriting the frozen `configs/chemeleon_stock/` and
  `results/chemeleon_stock/`. Make the stock-baseline label mode-aware (e.g. append the mode
  for the non-frozen protocols, or honor `--label-prefix`) so the three land in distinct
  config/result dirs, run `run_stock_baseline.sh` per protocol, and point each card's
  stock-baseline reference at the matching protocol's `chemeleon_stock` results. This adds two
  stock-baseline finetune runs (reduced, unlocked) on top of the existing frozen one.
  **All three code gaps are now wired (commit 4dd1ed1), data not yet generated.** (1)
  `report_card.py` gained `--lr-mode {reduced,unlocked}`: it filters `--metrics-csv` (point it
  at `results/lr_metrics.csv`) to that protocol's `lr_<mode>__<flavor>` rows and strips the
  prefix back to the bare flavor so the columns match the registry; the reference series still
  read the full frame, so pass the protocol's stock baseline via `--baseline-flavor
  chemeleon_stock_<mode>`. The frozen card is unchanged. (2) `meta_model.py` gained
  `--lr-mode`: it strips the `lr_<mode>__` prefix from each result-dir label at collection so
  the stacker's features match `flavor_names()` (the prefixed labels never matched before, so a
  reduced meta-model silently produced nothing) and writes a mode-scoped
  `meta_model_<estimator>_<mode>.csv`. (3) `generate.py`'s `stock_baseline_label` makes the
  stock label mode-aware (`chemeleon_stock` for frozen, `chemeleon_stock_<mode>` otherwise) so
  the protocols do not overwrite each other's config/result dirs. Regression tests added for
  all three. Still to do (submission orchestration, part of the 5-seed launch): finetune the
  stock baseline per protocol, run `meta_model.py --lr-mode` per protocol, and render the six
  cards (three protocols x two color modes).
- [x] 9. Meta-model: stack per-flavor finetuned predictions per endpoint, fit LGBM/RF/MLP
  on out-of-fold predictions, compare to the best single flavor.
  First real result, produced by the same job 19230968 now that ≥2 flavors have results:
  the LGBM meta-model beats the best single flavor on 23 of 32 endpoint-columns, mean
  R-squared 0.481 vs. 0.390 for the best single flavor per endpoint (mean delta +0.091).
  `rdkit2d` (11 endpoints) and `minimol` (9) are the most frequent single-flavor winners the
  meta-model has to beat; `usrcat` (4), `atompair`/`jazzy`/`e3fp` (2 each), `whim`/`ecfp` (1
  each) round out the rest. See `results/meta_model_lgbm.csv` for the per-endpoint table and
  `FINDINGS.md` for the read.
  **Per-protocol meta-models wanted for the final deliverable (see Milestone 8's note).** The
  full-corpus report cards need a meta-model per learning-rate protocol, not just frozen. Run
  `meta_model.py` three times, once against each protocol's mode-scoped prediction dirs, and
  write three distinct output CSVs (the default `meta_model_<estimator>.csv` name collides
  across runs); feed each into the matching protocol's two report cards.
- [ ] 10. (GATED on 8, and only if any flavor beats baseline) Scale the flavors that show
  utility up to the full 1M-molecule corpus to produce the final foundation-model artifacts.
  **Folded into Milestones 6-9's full-corpus rerun (Milestone 2's supersede note):** rather
  than screen on 250K first and scale only the winners, the whole sweep now runs directly on
  the full corpus, so every flavor's foundation is already the release-scale artifact once
  its pretrain/finetune completes. This milestone is void as originally scoped (there is no
  separate 250K-screen-then-scale step left to do); revisit only if a flavor needs to move
  beyond the current full corpus (944,296 molecules) to the full 1M CheMeleon PubChem set.

## Open items (need input or external data)

- [~] ML-QM flavor (dropped, 2026-07-08): the qmdesc-based flavor was implemented (24-dim
  pooled QM descriptors) and pretrained, but its target legitimately carries ~1.4% NaN
  (qmdesc fails on some molecules) and its full-corpus foundation came back all-NaN. Decision:
  do not run it; the flavor, its calculator, and `envs/mlqm.yml` are removed from the repo.
  See Milestone 7's drop note.
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
- [~] 5-seed finetune-only redo (in progress): replicate finetuning across 5 fresh seeds
  (1-5) off the fixed s42 foundations to put error bars on every report-card cell, without
  pretraining anything new. `generate.py` decouples the two seeds (commit 6a75165):
  `--foundation-seed` pins the foundation while `--seeds`/`--finetune-seed` vary the finetune,
  written into every `random_seed` field. The three runners take `FOUNDATION_SEED` to switch
  into finetune-only mode (skip corpus/target/prescale/pretrain, gate on the pinned foundations
  existing, submit only finetune + analyze): `run_all.sh` (flavors, frozen), `run_lr_experiments.sh`
  (flavors, reduced + unlocked), `run_ablations.sh` (ablations, all three protocols). Plan:
  flavors frozen 1800 tasks, flavors reduced+unlocked 3600, ablations x 3 protocols 2520.
  **Submitted so far:** flavors frozen only, `FOUNDATION_SEED=42 FLAVOR_SEEDS="1 2 3 4 5" bash
  slurm/run_all.sh` (finetune job 641124, array 0-1799; analyze job 641126 chained). The LR and
  ablation legs are wired but not yet launched (held pending the frozen leg completing).
- [x] **Blocker for Milestone 7, raised in urgency:** target-dropout fraction for small
  flavors. The masked-pretext dropout in `losses.py` (`DROPOUT_FRACTION`, applied per target
  element to every flavor) keeps a fixed fraction, not a fixed count. Its rationale (stop the
  head co-adapting across a wide descriptor block) is strong at 3585 dims (osmordred) but
  breaks down at low dims. Since the training-collapse regime fix, `DROPOUT_FRACTION=0.85`
  (keeps 15%, matching `../foundation-models/pretraining`'s `MASKING_RATIO`), so jazzy (6
  dims) now keeps under 1 target/step on average, worse than the previous 0.30 (~4 of 6
  kept) that this item originally flagged as merely noisy. Mechanically safe (loss aggregates
  over all kept elements in the batch, not per-row, so no divide-by-zero), but likely
  unusably sparse. Ablate the fraction (e.g. 0.0, 0.15, 0.85) per small flavor, holding the
  backbone and target fixed, the same way as the prescaling triage, **before** surrogate_adme
  (25 dims) fans out, not only "if they underperform" as originally
  scoped. Keep it fixed across the main sweep once decided; varying it per flavor mid-sweep
  would confound the report card.
  **Status: still open, now the reason surrogate_adme is held out of Milestone 7.**
  Milestone 6 shipped `jazzy` without this ablation (an explicit, recorded deferral for that
  one flavor); Milestone 7 does not repeat that deferral for `surrogate_adme` since
  this item names it directly. `minimol` (512 dims) is unaffected by this blocker and
  proceeds on its own.
  **Decision: override to 0.0 for the narrow flavors, skipping the ablation, not running it
  later.** Explicit call, not a default: the ablation (0.0 vs. 0.15 vs. 0.85) is not being run;
  0.0 is picked directly on the reasoning that under-1-target/step supervision is unlikely to
  beat no masking at all for a ~25-dim block, and confirmed post hoc rather than compared
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
  does not touch anything already on disk. To fan out `surrogate_adme` at 0.0,
  export `DROPOUT_FRACTION_OVERRIDES="surrogate_adme=0.0"` before submitting.
  **Superseded by an automatic rule, ahead of the full-corpus rerun.** The manual
  per-flavor `DROPOUT_FRACTION_OVERRIDES` list required remembering to export it for every
  narrow flavor; replaced with `config.py`'s `DROPOUT_OVERRIDE_MAX_DIM=30` and a
  `train.py` default that falls back to `dropout_fraction=0.0` for any target at or under
  that width once `n_features` is known from the split (jazzy at 6 and
  surrogate_adme at 25 qualify automatically; osmordred at 3585 and every other flavor
  keep the regime default of 0.85).
  **Hardened into an invariant: zero dropout below the threshold is not overridable.**
  This is a settled decision, not an open ablation: no flavor at or under
  `DROPOUT_OVERRIDE_MAX_DIM` may ever be pretrained with dropout. `train.py` now resolves the
  sub-threshold case first, before the `--dropout-fraction` flag, and raises a `SystemExit`
  if a nonzero `--dropout-fraction` (or `DROPOUT_FRACTION_OVERRIDES` entry) is passed for such
  a flavor rather than silently honoring it. The `--dropout-fraction` flag and
  `DROPOUT_FRACTION_OVERRIDES` list survive only to tune above-threshold flavors as a
  recorded one-off; they cannot raise dropout on a narrow flavor. The running full-corpus
  pretrain array already complies (jazzy/surrogate_adme all logged
  `dropout_fraction=0.000`, the PCA thresholds at 70/147/237 dims sit above the threshold at
  0.850), so nothing needed re-running. Do not reintroduce dropout on a sub-threshold flavor,
  raise the fraction for one, or lift `DROPOUT_OVERRIDE_MAX_DIM` to pull one back under
  masking. Mirrored in `AGENTS.md`'s experiment-discipline invariants and the methodology
  watch-item below.
- [ ] Frozen warmup then coadaptation: train for N epochs with `mpnn_lr=0` so the FFN head
  finds a reasonable operating point against the fixed representations, then unfreeze the
  MPNN and continue training at a reduced rate. Avoids the large gradient shock that occurs
  when a randomly initialized head immediately backpropagates into a pretrained backbone,
  while still allowing the MPNN and FFN to coadapt once the head has stabilized. Requires
  a two-phase training schedule not currently supported by the anvil config; likely needs a
  custom Lightning callback or a sequential two-recipe approach. This is the one LR experiment
  `run_lr_experiments.sh` does not cover (it only sweeps single-rate protocols: reduced,
  unlocked); wiring it needs that anvil feature first.
- [ ] PCA-compressed osmordred target (wiring in progress, not yet run): osmordred only, backbone/corpus/regime
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
  analyze plumbing.
  **No longer plan-only; wiring underway alongside the full-corpus rerun.** `flavors.py`
  gained a `derived_from` field (a flavor whose target has no calculator of its own, built
  instead from a base flavor's already-computed target) and registered
  `osmordred_pca80`/`pca90`/`pca95`; `sarizard/pretraining/pca_target.py` fits PCA once on
  the `full`-recipe-prescaled osmordred store (train rows only) to the largest requested
  threshold and slices a prefix of the fitted components for the smaller thresholds;
  `slurm/osmordred_pca_targets.sbatch` runs it, gated into `run_all.sh`'s chain only when
  the registry has a derived flavor. `compute_targets.sbatch` and `split.sbatch` skip a
  derived flavor's own target/prescale stage and treat its PCA output as already prescaled.
  Recipes are generated (`configs/osmordred_pca{80,90,95}__s42/`); its target-derivation
  stage is job 33588 in the full-corpus `run_all.sh` chain described in Milestone 6, and
  pretrain/finetune ride the same chain (jobs 33590/33591) once split completes. **Pretrain
  is now complete: all three `osmordred_pca80/90/95` foundations are on disk (pca90 via the
  93729_14 resubmit after a transient GPU failure); finetune still pending, see Milestone 6.**

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
  training-collapse regime fix, keeps 15%) is a fixed fraction applied to every
  above-threshold flavor, so its effect scales with target width: reasonable supervision
  density at high dims (osmordred, 3585), unusably sparse at low dims (jazzy 6, surrogate_adme 25,
  under 1 target/step on average). That sparsity is resolved, not open: any target at or
  under `config.DROPOUT_OVERRIDE_MAX_DIM` (30 dims) pretrains at `dropout_fraction=0.0` as a
  hard invariant enforced in `train.py`, which rejects a nonzero override for such a flavor.
  Do not reintroduce dropout on a sub-threshold flavor, raise its fraction, or lift the
  threshold to bring one back under masking; see the resolved blocker in Future experiments
  and the invariant in `AGENTS.md`.
- **Finetuning always covers all three learning-rate protocols (standing directive).** Any
  finetuning submission runs frozen (`mpnn_lr=0`), reduced (`mpnn_lr=ffn_lr/10`), and
  unlocked (`mpnn_lr=ffn_lr`), not frozen alone. `run_all.sh`'s finetune stage is frozen-only
  by construction, so pair it with `slurm/run_lr_experiments.sh` (`LR_MODES="reduced
  unlocked"`, `FLAVOR_SEEDS` matching the sweep) off the same foundations; the LR path reuses
  the pretrained foundations, so submit it only after every foundation the sweep needs is on
  disk. Do not treat "finetune" as frozen-only.
- Pretraining regime constants (`sarizard/pretraining/config.py`) are reconciled against the
  sibling `../foundation-models/pretraining` implementation as of the training-collapse fix
  (`PATIENCE`, `FNN_HIDDEN_SIZE`, `WARMUP_EPOCHS`, `DROPOUT_FRACTION`, `GRADIENT_CLIP_VAL`,
  precision). Treat that sibling as the reference for any future regime question; a
  divergence from it is now a deliberate choice, not an oversight, and should be commented
  as such.
