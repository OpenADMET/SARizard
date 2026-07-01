# Findings: fit-to-purpose CheMeleon foundations

Source of record for headline results. Per-flavor and per-endpoint detail lives in the
Obsidian wiki under `wiki/`; where the wiki and this file disagree, this file wins.

## The question

Does the descriptor target a CheMeleon-style foundation is pretrained against determine
which ADMET endpoints and endpoint families it serves best? And does stacking the
per-flavor finetuned predictions into a meta-model beat the best single foundation?

## Status

Prescaling ablation triage complete (see below): the winning recipe differs by finetune
protocol, so the Milestone-5 recipe pick is the open decision blocking the flavor sweep.
No flavor-sweep results yet; the report card follows once that decision lands and the
sweep (Milestone 6) runs.

## Prescaling

**Superseded.** Every pretraining run behind the 250K numbers below diverged mid-training
(val/R2 and val_loss blowing up by 2-6 orders of magnitude within a single epoch, 4-10
epochs in depending on recipe); the "winner" in each protocol below is confounded by which
recipe's trajectory happened to survive longest before collapsing, not by prescaling
quality. Root cause and fix: see "Training collapse and regime fix" below. The 250K
artifacts (targets, splits, foundations, configs, results, plots) are archived at
`archive/ablation_250k_pre_gradclip/`; do not use the numbers below to pick the production
recipe. A rerun on the full corpus with the fixed regime is in progress; its numbers will
replace this section once complete.

### 250K numbers (historical, do not use for the Milestone-5 decision)

The ablation triage (osmordred, 7 recipes, seed 42) is finetuned and evaluated under all
three MPNN-LR protocols (frozen, reduced, unlocked); `archive/ablation_250k_pre_gradclip/results/ablation_metrics.csv`
(672 rows) holds the full numbers.

The winning recipe is not the same across protocols:

- **frozen** (the flavor sweep's protocol): `plus_yeo_johnson` wins, mean R-squared 0.320
  vs. `chemeleon_baseline` 0.292 (+0.028, +9.5% relative). `order_fix` and `full` are close
  behind (0.308, 0.310); `minimal` (no winsorization) is worst (0.256), so winsorizing
  before z-scoring matters most when the backbone is frozen.
- **reduced** (`mpnn_lr=1e-4`): `order_fix` wins, 0.350 vs. baseline 0.302 (+0.047, +15.7%),
  the largest margin of the three protocols. `plus_drop_low_var` and `plus_yeo_johnson` are
  close seconds (0.345, 0.343).
- **unlocked** (`mpnn_lr=1e-3`): `plus_drop_corr` wins, 0.304 vs. baseline 0.295 (+0.009,
  +3.1%), the smallest margin. Once the MPNN backbone can move freely, prescaling choice
  barely matters, the model adapts around it.

Read: prescaling's payoff shrinks as the backbone unfreezes, consistent with prescaling
mainly compensating for what a frozen backbone/FFN head cannot fix on its own. `order_fix`
(winsorize before z-score, the minimal correction to today's `split.py` order bug) is the
most consistent performer, never worse than 2nd-3rd in any protocol, while `plus_yeo_johnson`
wins outright under frozen, the protocol the core flavor sweep actually uses.

Milestone-5 decision (recipe to bake into `split.py`): pending, and now moot until the
rerun lands, since these numbers are confounded by the training collapse (above).

### Training collapse and regime fix

TensorBoard curves for all 7 250K ablation runs showed the same failure: smooth, sane
val/R2 climbing toward 0.85-0.93, then a catastrophic blowup to large negative R2 in one
epoch, with no recovery within the early-stopping patience window. Auditing
`../foundation-models/pretraining/run_pretraining.py` (same MPNN/descriptor-regression task,
no such instability observed) found no gradient clipping in either implementation, but three
real regime departures: `PATIENCE` 5 vs. the sibling's 50, predictor width 2048 vs. 1024,
and the masked-pretext keep fraction 70% (`DROPOUT_FRACTION=0.30`) vs. the sibling's 15%
(a much denser per-step supervision load on the 3585-dim osmordred target). The sibling also
trains bf16-mixed precision where SARizard trained full fp32.

Adopted the sibling's regime as canonical (`sarizard/pretraining/config.py`: `PATIENCE=50`,
`FNN_HIDDEN_SIZE=1024`, `WARMUP_EPOCHS=2`, `DROPOUT_FRACTION=0.85`), added
`GRADIENT_CLIP_VAL=0.5` and bf16/16-mixed precision on top (neither implementation had
these), and moved the triage to the full corpus (`corpus/corpus_full.parquet`, ~900K
molecules) rather than paying for 4x the epochs on 250K, eliminating corpus size as a
confound at the same time as the regime fix. Rerunning `chemeleon_baseline` alone first to
confirm stability before firing the other six recipes.

Corpus build landed (944296 molecules; job 18097840). The first full-corpus osmordred target
job (18097911) failed immediately on an unrelated infrastructure fault: the `sarizard-osmordred`
conda env was torn (missing `libparquet` shared libraries despite conda-meta listing the
package installed, and unregistered with `conda env list`), left over from a prior `setup.sh`
run that did not complete. Rebuilt clean via `FORCE=1 bash setup.sh osmordred`; target
computation resubmitted and completed (job 18106135).

The next `chemeleon_baseline` prescale+split (job 18106233) and pretrain (job 18106234) were
submitted without `CORPUS_FILE`/`CORPUS_N` in the environment, so `slurm/env.sh` defaulted
`CORPUS_FILE` back to the 250K corpus. `ablation_prescale.sbatch` prescaled the correct
full-corpus target (it reads `cache/targets/osmordred/target.zarr`, already computed on the
full corpus), but `split.py`'s `--input-smiles` then pointed at the 250K corpus parquet, so
`train_smiles.parquet` held 224144 (250K-derived) rows against a full-corpus-sized
`train_rescaled.zarr` (849920 rows), a silent row-count divergence between the target and its
SMILES that pretrain's dataset loader caught (`smiles/target row mismatch`) before any training
step ran. Deleted the stale `cache/ablations/chemeleon_baseline/split/` directory (the
`prescaled.zarr` above it was unaffected) and resubmitted prescale (job 18108226) and pretrain
(job 18108227, dependent on it) with `CORPUS_FILE=corpus/corpus_full.parquet CORPUS_N=1000000`
exported in the submitting shell.

Prescale (18108226) completed clean this time, but pretrain (18108227) failed 14s in on a
second, unrelated gap: the `sarizard` env was missing `tensorboard`, which `envs/main.yml`
already declares (`tensorboard>=2.15`, with a comment noting Lightning's `TensorBoardLogger`
needs it) but which was never installed, an env left stale relative to its own spec rather
than anything touched by this triage. Fixed with `conda env update -n sarizard -f
envs/main.yml` (also picked up a chemprop patch bump, 2.2.3 to 2.2.4, within its `>=` pin; test
suite still passes). Pretrain resubmitted, job 18108864, and started training, but Lightning
reported `GPU available: False, used: False` and ran on CPU despite the `--gres=gpu:1`
allocation: a third, pre-existing gap, `sarizard`'s PyTorch was the conda-forge CPU-only build
(`cpu_mkl_py311`, `torch.version.cuda` is `None`). `envs/main.yml` deliberately excludes a
PyTorch build from its generic spec (`pytorch>=2.2` only) with a comment to install a
CUDA-matched build by hand per cluster; that step was never done for this env. Cancelled
18108864 before it burned more GPU-allocated wall clock on CPU, installed
`pytorch=2.12.0=cuda129_mkl_py311*` (CUDA 12.9, matching the A40/A100 nodes' driver 590.48),
confirmed `torch.cuda.is_available()` is `True` and the test suite still passes, and
resubmitted. Job 18109452 confirms `GPU available: True (cuda), used: True` on an A100 80GB.

`chemeleon_baseline` ran clean through 15 epochs: `val/r2` climbed 0.773 to 0.952 (one small
dip to 0.852 at epoch 6, recovered the next epoch, ordinary noise, not a collapse), `val_loss`
and `train_loss` both monotonic down throughout, well past the epoch 4-10 window where all
seven 250K runs previously diverged. Regime fix confirmed stable; submitted the remaining six
prescaling recipes (`minimal`, `order_fix`, `plus_drop_corr`, `plus_drop_low_var`,
`plus_yeo_johnson`, `full`; prescale job 18111455, pretrain job 18111456 dependent on it).

One consequence to flag: `DROPOUT_FRACTION=0.85` (keep 15%) is a fixed-regime constant
applied to every future flavor. For osmordred (3585 dims) that is ~537 supervised dims/step,
fine; for a small flavor like jazzy (6 dims) later, that is under 1 supervised dim/step on
average, worse than the borderline case already flagged in `TODO.md`. Flagged there as a
pre-Milestone-6 item rather than fixed now, per the fixed-regime discipline (do not
special-case flavors mid-sweep).

## Report card

To be filled. The artifact is `sarizard/analysis/report_card.py`: rows are endpoints across all
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
