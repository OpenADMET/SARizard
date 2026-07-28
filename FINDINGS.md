# Findings: fit-to-purpose CheMeleon foundations

Source of record for headline results. Per-flavor and per-endpoint detail lives in the
Obsidian wiki under `wiki/`; where the wiki and this file disagree, this file wins.

## The question

Does the descriptor target a CheMeleon-style foundation is pretrained against determine
which ADMET endpoints and endpoint families it serves best? And does stacking the
per-flavor finetuned predictions into a meta-model beat the best single foundation?

## Status

**Current (2026-07-27): every planned run is finished and nothing is queued.** The full-corpus
flavor sweep covers all 15 flavors under all three finetune protocols at 5 finetune seeds each,
against a 5-seed stock-CheMeleon baseline per protocol, and all six report cards are rendered.
Four side studies are also complete: the 5-seed prescaling ablation redo, the
external-foundation comparison, the `osmordred_surrogate` chemical-space control, and the PXR
external-test rerun. Each has its own section below.

The headline answers as they now stand:

- **Descriptor target matters, but less than the finetune protocol.** Under frozen and reduced,
  the learned-model flavors (`surrogate_adme`, `minimol`) and `rdkit2d` clear the stock baseline
  significantly; the binary fingerprints fall well below it. Under unlocked, almost everything
  collapses back to or below stock, so the pretraining target only shows through while the
  backbone is held still or nearly still.
- **Specialization is real but narrow.** The per-endpoint winner changes across families, but no
  flavor beats stock on an entire family the way the fit-to-purpose premise would predict, and
  the PXR external test shows the one clean specialization signal (`rdkit2d` on the internal
  Butina split) does not survive a fixed external hold-out.
- **Stacking beats any single foundation.** The LGBM meta-model wins 21 of 32 endpoint-columns
  (mean delta R-squared +0.082 over the best single flavor per endpoint).
- **Pretraining corpus size is not the lever.** Four externally pretrained foundations, spanning
  1M to 10M molecules, all lose to stock CheMeleon under every protocol, significantly.

Only one step from the original plan remains unrun: the per-mode (reduced/unlocked) meta-models,
which are wired but never launched, so `results/meta_model_lgbm.csv` is frozen-only. One gap is
worth naming rather than closing silently: the Milestone-5 prescaling decision rests on
single-seed data that the 5-seed redo did not re-test (see Prescaling below).

The rest of this section is the prior history leading here.

Prescaling ablation triage complete on the full corpus with the regime fix, including the
cross-protocol check (frozen/reduced/unlocked; see below): `chemeleon_baseline` wins under
frozen, the protocol the flavor sweep uses, though by a close margin over `order_fix`, which
wins under the other two protocols. Since `chemeleon_baseline` is already what `split.py`
produces today, Milestone 5 needs no code change, just the recorded decision below. A
follow-up 250K corpus-size check (same 7 recipes x 3 protocols) found a different ranking
entirely at the smaller scale, confirming corpus size itself shapes which recipe wins; see
below. This does not change the Milestone-5 decision, since the flavor sweep runs on the
full corpus. Milestone 6's direct-compute flavor sweep (rdkit2d, erg, ecfp, atompair,
pubchem, usrcat, whim, e3fp, jazzy; 250K corpus, `order_fix` prescaling, all three
finetune protocols) completed with no failures; see the report card below. Milestone 7
(learned-model flavors: minimol, surrogate_adme) first ran on 250K scoped to
`minimol` only, with `surrogate_adme` held out over the target-dropout-fraction
question. That question is now settled by a hard invariant (surrogate_adme is sub-threshold, so it
pretrains at `dropout_fraction=0.0`; see `TODO.md` Future experiments and `AGENTS.md`), and
both fan out alongside every other flavor in the full-corpus rerun. `minimol`'s 250K finetune
chain (frozen protocol)
completed clean: mean R-squared 0.360 across 32 endpoint-columns, edging out `rdkit2d`'s
0.350 frozen mean from Milestone 6, though the two were not evaluated in one merged table
so this is context, not a controlled comparison yet. The chained analyze job exited
nonzero, but only because `meta_model.py` deliberately refuses to stack fewer than two
flavors; the report-card stage before it wrote `results/metrics.csv` and the plots
successfully. See `TODO.md` Milestone 7 for the full account, including a note that this
run overwrote Milestone 6's merged `results/metrics.csv` (recoverable from cached
predictions; nothing was lost).

## Prescaling

**Milestone-5 decision: `chemeleon_baseline`, no code change needed.** The full-corpus
rerun under the regime fix (below) ranks `chemeleon_baseline` first by mean R-squared under
the frozen protocol, the protocol the core flavor sweep uses. `split.py` already reproduces
`chemeleon_baseline` today (mean/std computed on the raw target, reused for both
winsorization and z-scoring), so Milestone 5 (bake the winning recipe into `split.py`) is
satisfied as-is; the flavor sweep (Milestone 6) can proceed on the current `split.py` path
unmodified. The margin over the runner-up (`order_fix`) is close, not decisive; see the
cross-protocol check below for the full picture.

### Full-corpus numbers (regime-fixed, decision basis)

`osmordred`, 7 recipes, seed 42, full corpus (`corpus/corpus_full.parquet`, 944296
molecules), frozen protocol only (`mpnn_lr=0`, the flavor sweep's protocol).
`results/ablation_metrics.csv` (224 rows) and `plots/prescaling_report_r2.csv` hold the full
numbers. Mean R-squared across the 24 endpoints:

| recipe | mean R-squared |
|---|---|
| `chemeleon_baseline` | 0.352 |
| `order_fix` | 0.342 |
| `plus_yeo_johnson` | 0.340 |
| `plus_drop_low_var` | 0.327 |
| `full` | 0.326 |
| `plus_drop_corr` | 0.324 |
| `minimal` | 0.307 |

Read: today's entangled winsorize/z-score order (`chemeleon_baseline`) is not a bug worth
fixing, at least not under the frozen protocol. Every attempted correction or addition
(`order_fix` through `full`) trails it by 1-3 points of R-squared, and `minimal` (no
winsorization) is clearly worst, so winsorizing before z-scoring matters, but the specific
order/stat-sharing quirk in the production path does not hurt downstream transfer enough to
justify the extra pipeline steps. This reverses the historical 250K read below
(`plus_yeo_johnson` winning under frozen), which is expected since that run was confounded
by the training collapse, not a real signal.

### 5-seed redo, and the gap it leaves in the decision

Every prescaling number in the sections below is single-seed. The triage was later re-run at 5
finetune seeds off the same fixed s42 ablation foundations, across all three protocols (2160
finetunes); `results/ablation_metrics.csv` and the per-protocol
`plots/ablation_report_card_r2[_<mode>].png` cards hold it. Mean R-squared across the 24
ablation endpoints, per seed then averaged over seeds 1-5:

| recipe | frozen | reduced | unlocked |
|---|---|---|---|
| plus_yeo_johnson | **0.313 +/- 0.009** | 0.354 +/- 0.006 | 0.295 +/- 0.020 |
| plus_drop_low_var | 0.310 +/- 0.015 | **0.356 +/- 0.008** | **0.312 +/- 0.014** |
| plus_drop_corr | 0.299 +/- 0.013 | 0.350 +/- 0.014 | 0.304 +/- 0.031 |
| full | 0.298 +/- 0.011 | 0.341 +/- 0.011 | 0.297 +/- 0.018 |
| order_fix | 0.294 +/- 0.007 | 0.347 +/- 0.008 | 0.307 +/- 0.026 |
| minimal | 0.286 +/- 0.016 | 0.336 +/- 0.020 | 0.293 +/- 0.015 |
| chemeleon_stock (reference) | 0.295 +/- 0.009 | 0.316 +/- 0.014 | 0.337 +/- 0.008 |

**`chemeleon_baseline`, the recipe the Milestone-5 decision names, is not in this table.** The
redo covers six recipes, not seven: `chemeleon_baseline` kept only its single-seed s42 result
dirs, its configs were cleared along with the other stale s42 recipes, and it never entered
`results/ablation_metrics.csv`. So the 5-seed data can neither confirm nor overturn the ranking
that selected it.

What the six that were re-run do show is that the single-seed frozen ordering did not survive
seed averaging: `plus_yeo_johnson` now leads frozen (it was 3rd of 7 single-seed) and
`plus_drop_low_var` leads both non-frozen protocols (it was the single-seed bottom performer,
eliminated first in every ranked-choice round). The recipes also sit within roughly one seed
standard deviation of each other under frozen, which is the more useful read: at 5 seeds the
prescaling choice is close to noise under the protocol the sweep uses.

This does not destabilize anything operationally. `split.py` already produces
`chemeleon_baseline`, so revisiting the decision would change a recorded rationale, not code.
But the recorded margin should be read as single-seed, and closing the gap properly means
finetuning `chemeleon_baseline` at seeds 1-5 across the three protocols (360 recipes off the
existing foundation, no pretraining).

### Cross-protocol check (reduced/unlocked, full corpus, regime-fixed, single-seed)

The frozen-only result above was later crossed with the two other finetune protocols
(`reduced`, `mpnn_lr=1e-4`; `unlocked`, `mpnn_lr=1e-3`) on the same seven full-corpus
foundations, closing the gap this section used to flag. 504 finetune runs total (7 recipes
x 3 protocols x 24 endpoints); `results/ablation_metrics.csv` (672 rows) and
`plots/prescaling_mode_comparison_r2.csv` hold the numbers.

| recipe | frozen | reduced | unlocked |
|---|---|---|---|
| minimal | 0.3073 | 0.3341 | 0.3077 |
| chemeleon_baseline | **0.3523** | 0.3772 | 0.3041 |
| order_fix | 0.3415 | **0.3884** | 0.3119 |
| plus_drop_corr | 0.3236 | 0.3709 | **0.3218** |
| plus_drop_low_var | 0.3269 | 0.3582 | 0.2820 |
| plus_yeo_johnson | 0.3404 | 0.3666 | 0.3171 |
| full | 0.3263 | 0.3813 | 0.3013 |

The winning recipe shifts by protocol: `chemeleon_baseline` wins frozen, `order_fix` wins
reduced, `plus_drop_corr` wins unlocked. `reduced` is uniformly the best protocol for every
recipe, so a little backbone movement helps regardless of prescaling; `unlocked` compresses
the spread between recipes (0.28-0.32) and is the only protocol where `minimal` (no
winsorization) is competitive rather than clearly worst, consistent with the 250K-era read
that prescaling matters less once the backbone can fully adapt.

Per-endpoint win counts (best recipe per endpoint, 29 endpoints per protocol, ablation
recipes only) tell a similar but not identical story:

| recipe | frozen | reduced | unlocked | total |
|---|---|---|---|---|
| chemeleon_baseline | 3 | 6 | 6 | 15 |
| order_fix | 2 | 7 | 5 | 14 |
| plus_drop_corr | 4 | 4 | 6 | 14 |
| minimal | 4 | 2 | 5 | 11 |
| plus_yeo_johnson | 1 | 6 | 4 | 11 |
| full | 2 | 3 | 1 | 6 |
| plus_drop_low_var | 2 | 1 | 2 | 5 |

Treating each endpoint's full R-squared ranking as a ranked-choice ballot and running
instant-runoff (eliminate the recipe with fewest first-place endpoints each round,
redistribute to those endpoints' next-best recipe) gives a third view that rewards
consistent strength over occasional spikes: `order_fix` wins frozen, `chemeleon_baseline`
wins reduced, `minimal` wins unlocked. Pooling all three protocols into one 87-ballot
election, `order_fix` wins, but by a single vote in the final round (44 to 43 over
`chemeleon_baseline`); `chemeleon_baseline` actually led every round from round 1 through
round 5, and only fell behind once `plus_yeo_johnson`'s 27 eliminated ballots redistributed
in `order_fix`'s favor. `plus_drop_low_var` is eliminated first in every protocol and in the
pooled election, confirming it as the clear bottom performer across every read.

Read: `chemeleon_baseline` and `order_fix` are statistically close, not a clean margin.
`chemeleon_baseline` wins the mean-R-squared and win-count tallies under frozen (the only
protocol the flavor sweep actually uses) and led the pooled ranked-choice election for most
of its rounds; `order_fix` wins the pooled ranked-choice election outright and the mean-R-
squared ranking under reduced. The Milestone-5 decision (`chemeleon_baseline`, recorded
below and in `TODO.md`) stands because it is the frozen-protocol winner and frozen is what
the sweep runs, but it should be read as "the better of two very similar recipes," not a
decisive win; `order_fix` is the natural second read if the sweep protocol ever changes.

### 250K corpus-size check (regime-fixed, valid)

Repeats the same 7 recipes x 3 protocols triage on the original 250K screening corpus
(`corpus/corpus_250k.parquet`), now that the regime fix makes a 250K run valid (unlike the
pre-fix numbers below). Tests whether the full-corpus ranking holds at 1/4 the corpus size,
or whether corpus size itself changes which recipe wins. Full-corpus artifacts were archived
to `archive/ablation_full_corpus/` first so this run does not overwrite them (mirroring the
`archive/ablation_250k_pre_gradclip/` precedent). All 504 finetunes and the chained analyze
completed clean; `results/ablation_metrics.csv` (672 rows) and
`plots/prescaling_mode_comparison_r2.csv` hold the numbers.

| recipe | frozen | reduced | unlocked |
|---|---|---|---|
| minimal | 0.2729 | 0.3624 | **0.3244** |
| chemeleon_baseline | 0.2951 | 0.3607 | 0.3233 |
| order_fix | **0.3407** | 0.3378 | 0.3188 |
| plus_drop_corr | 0.3219 | 0.3355 | 0.2865 |
| plus_drop_low_var | **0.3465** | 0.3746 | 0.3041 |
| plus_yeo_johnson | 0.3222 | **0.3808** | 0.2735 |
| full | 0.3223 | 0.3173 | 0.2764 |

**Result: the 250K ranking does not match the full-corpus ranking.** Winners are
`plus_drop_low_var` (frozen), `plus_yeo_johnson` (reduced), `minimal` (unlocked); none match
their full-corpus counterparts (`chemeleon_baseline`, `order_fix`, `plus_drop_corr`).
`chemeleon_baseline`, the full-corpus frozen winner and the Milestone-5 decision, drops to
5th of 7 under frozen at 250K.

Comparing mean R-squared (averaged across the three protocols) side by side:

| recipe | 250K mean | full mean | delta (full − 250K) |
|---|---|---|---|
| minimal | 0.3199 | 0.3164 | -0.0035 |
| chemeleon_baseline | 0.3264 | 0.3445 | +0.0182 |
| order_fix | 0.3324 | 0.3473 | +0.0148 |
| plus_drop_corr | 0.3146 | 0.3388 | +0.0241 |
| plus_drop_low_var | 0.3417 | 0.3224 | -0.0194 |
| plus_yeo_johnson | 0.3255 | 0.3414 | +0.0159 |
| full | 0.3053 | 0.3363 | +0.0310 |

Most recipes score higher on the full corpus, as expected ("more data helps"), led by `full`
(+0.031) and `plus_drop_corr` (+0.024). But `plus_drop_low_var` and `minimal` score *lower*
on the full corpus (-0.019, -0.004), which isn't the "more data always helps" direction.
`chemeleon_baseline` gains a real +0.018 on the full corpus but starts from a mediocre 250K
position (4th of 7), meaning it is not a strong recipe at small scale, just one that scales
unusually well.

A pooled ranked-choice election (all three 250K protocols, 87 ballots) is far more decisive
here than the full-corpus equivalent: `chemeleon_baseline` leads every single round and wins
the final round 48 to 39 over `plus_yeo_johnson`, a clean margin (compare the full-corpus
pooled election, a 44-43 squeaker for `order_fix`). Per-protocol IRV winners are
`plus_drop_low_var` (frozen), `plus_drop_low_var` (reduced, despite `plus_yeo_johnson`
winning on mean R-squared, the same win-count-vs-mean divergence seen on the full corpus),
and `chemeleon_baseline` (unlocked).

Read: corpus size is doing real work in which prescaling recipe wins, not just adding
statistical power to the same ranking. Does not change the Milestone-5 decision as recorded
(the flavor sweep runs on the full corpus, so the full-corpus frozen ranking is the relevant
one), but the recipe ranking should not be assumed to generalize to a different corpus size,
e.g. if the full 1M corpus is ever substituted in for milestone 10.

### 250K numbers, pre-regime-fix (historical, superseded, do not use)

Not to be confused with the valid 250K corpus-size check above, run after the regime fix.
Every pretraining run behind the numbers below diverged mid-training (val/R2 and val_loss
blowing up by 2-6 orders of magnitude within a single epoch, 4-10 epochs in depending on
recipe); the "winner" in each protocol is confounded by which recipe's trajectory happened
to survive longest before collapsing, not by prescaling quality. Root cause and fix: see
"Training collapse and regime fix" below. The 250K artifacts (targets, splits, foundations,
configs, results, plots) are archived at `archive/ablation_250k_pre_gradclip/`; kept for
reference only.

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

Milestone-5 decision (recipe to bake into `split.py`): superseded by the full-corpus rerun
above (`chemeleon_baseline`); these numbers were confounded by the training collapse.

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
`plus_yeo_johnson`, `full`; prescale job 18111455, pretrain job 18111456 dependent on it). All
seven pretraining runs completed clean (full 100 epochs, stable losses, no collapse).

Finetuning hit one unrelated infrastructure gap: the `openadmet` conda env had `boto3`
1.43.37 paired with a mismatched `botocore` 1.43.0, so every `openadmet anvil` invocation
failed at CLI import time (`ImportError: cannot import name 'DocumentModifiedShape' from
'botocore.docs.utils'`) before any recipe logic ran, killing all 168 finetune tasks (job
18410392). Fixed with `pip install --upgrade --force-reinstall boto3` in the `openadmet` env
(resolved both packages to 1.43.39), verified with a direct `openadmet anvil` run before
resubmitting. Rerun (job 18420137) completed all 168 finetunes clean; the chained analyze
job (18420145) produced `results/ablation_metrics.csv` and `plots/prescaling_report_r2.csv`,
the full-corpus numbers above.

One consequence to flag: `DROPOUT_FRACTION=0.85` (keep 15%) is a fixed-regime constant
applied to every above-threshold flavor. For osmordred (3585 dims) that is ~537 supervised
dims/step, fine; for a small flavor like jazzy (6 dims) that is under 1 supervised dim/step
on average. This is now resolved by a hard invariant, not left open: any target at or under
`config.DROPOUT_OVERRIDE_MAX_DIM` (30 dims: jazzy 6, surrogate_adme 25) pretrains at
`dropout_fraction=0.0`, enforced in `train.py` before the `--dropout-fraction` flag and not
overridable to a nonzero value. See the resolved blocker in `TODO.md` and the invariant in
`AGENTS.md`.

## Report card

### Full-corpus sweep, 5 seeds, all three protocols (current headline)

All 15 flavors on the full corpus (`corpus/corpus_full.parquet`, 944,296 molecules), finetuned
at 5 seeds (1-5) off the fixed seed-42 foundations under each of frozen, reduced, and unlocked,
against a 5-seed stock-CheMeleon baseline per protocol. `results/metrics.csv` (frozen) and
`results/lr_metrics.csv` (reduced, unlocked) hold the rows; the six cards are
`plots/report_card_{r2,mae_delta}[_reduced,_unlocked}.png`.

Each cell is the mean R-squared across the 32 endpoint-columns within a seed, then averaged over
the 5 seeds, plus or minus the seed standard deviation. Bold marks a flavor significantly above
its own protocol's stock baseline, "(below)" one significantly beneath it. The 15 flavors within
a protocol are one comparison family, tested together with Dunnett's test at p at or below 0.05
(see Multiple comparisons); these are family-wise verdicts, not 45 independent tests. Sorted by
the frozen column:

| Flavor | Frozen | Reduced | Unlocked |
|---|---|---|---|
| surrogate_adme | **0.370 +/- 0.011** | **0.374 +/- 0.023** | 0.368 +/- 0.016 |
| minimol | **0.343 +/- 0.006** | **0.371 +/- 0.020** | 0.308 +/- 0.020 |
| rdkit2d | **0.323 +/- 0.011** | **0.356 +/- 0.013** | 0.324 +/- 0.026 |
| osmordred_pca80 | 0.315 +/- 0.010 | 0.338 +/- 0.009 | 0.304 +/- 0.014 |
| osmordred_pca90 | 0.315 +/- 0.022 | 0.336 +/- 0.008 | 0.287 +/- 0.025 (below) |
| jazzy | 0.305 +/- 0.009 | 0.343 +/- 0.008 | 0.322 +/- 0.018 |
| osmordred_pca95 | 0.304 +/- 0.009 | **0.345 +/- 0.019** | 0.283 +/- 0.014 (below) |
| osmordred | 0.301 +/- 0.013 | 0.341 +/- 0.010 | 0.298 +/- 0.022 (below) |
| chemeleon_stock (reference) | 0.294 +/- 0.010 | 0.316 +/- 0.014 | 0.337 +/- 0.008 |
| usrcat | 0.285 +/- 0.018 | 0.314 +/- 0.018 | 0.313 +/- 0.017 |
| atompair | 0.270 +/- 0.011 | 0.310 +/- 0.007 | 0.303 +/- 0.028 |
| pubchem | 0.270 +/- 0.014 | 0.303 +/- 0.007 | 0.306 +/- 0.012 |
| erg | 0.265 +/- 0.005 (below) | 0.322 +/- 0.015 | 0.303 +/- 0.026 |
| e3fp | 0.233 +/- 0.008 (below) | 0.254 +/- 0.006 (below) | 0.263 +/- 0.011 (below) |
| ecfp | 0.214 +/- 0.020 (below) | 0.253 +/- 0.022 (below) | 0.243 +/- 0.016 (below) |
| whim | 0.210 +/- 0.024 (below) | 0.276 +/- 0.027 (below) | 0.304 +/- 0.021 |

Three reads come out of this table.

**Reduced is the protocol where pretraining pays, and the effect is narrower than it first
looked.** Four flavors clear the stock baseline significantly under reduced against three under
frozen, and the same three lead both. Under unlocked the picture inverts: the stock baseline
(0.337) is the best column on the card, **no flavor clears it significantly**, and five fall
significantly below. Letting the backbone move at the full head learning rate overwrites
whatever the descriptor pretraining installed, which is what a protocol that retrains the
representation should do.

Much of the apparent structure here was multiplicity. Before correcting, this table showed 10,
11 and 12 significant flavors per protocol; 7, 7 and 5 survive, and `surrogate_adme`'s unlocked
lead over stock (+0.032, family-wise p=0.11) does not survive at all. The flavors that drop out
are the ones that were marginal to begin with: `osmordred` and the three PCA variants under
frozen and reduced, `atompair` and `pubchem`'s frozen deficits, `jazzy`'s reduced margin.

**The learned-model flavors lead, and one of them is not a fair column.** `surrogate_adme` tops
all three protocols, but it pretrains on its own Novartis corpus per the AGENTS.md invariant, so
it is a different-corpus reference arm rather than an apples-to-apples column. The
`osmordred_surrogate` control (below) attributes most of its lead to its on-task ADME target
rather than its chemical space. `minimol`, which does share the corpus, is the strongest
legitimate flavor under frozen and reduced.

**The binary fingerprint flavors are consistently worst.** `ecfp` and `e3fp` sit significantly
below stock under all three protocols, and `whim` under frozen and reduced; `atompair` and
`pubchem` trail without separating significantly once corrected. This is
the leaky-and-weak-pretext prior in the methodology watch-items showing up as a result: a target
the message-passing network can read off the graph deterministically teaches it little.

Note that these numbers are lower across the board than the single-seed seed-42 table this
section used to carry (which had `surrogate_adme` 0.369, `osmordred` 0.327, stock 0.297). That
table was one finetune seed per flavor. The 5-seed averages are the honest version; the ordering
of the top flavors survived, the middle of the table reshuffled inside its error bars, and the
lesson is the same one the multi-seed baseline work below found on PXR.

#### PCA-compressed osmordred targets

`osmordred_pca80`, `pca90`, and `pca95` pretrain against PCA component scores fitted on the
`full`-recipe-prescaled osmordred matrix (train rows only) rather than the raw 3585-dim block:
70, 147, and 237 components respectively (`cache/targets/osmordred_pca_summary.json`). All three
land within one seed standard deviation of full `osmordred` under every protocol, and the three
thresholds do not order consistently with each other (pca80 leads frozen and unlocked, pca95
leads reduced), so the spread between them is seed noise rather than an explained-variance
effect.

Compression is therefore free, not a win: a 15-to-50x narrower, decorrelated target trains an
equally transferable foundation for a fraction of the pretraining cost, but not a better one.
The secondary motivation, that a narrow target changes the masked-pretext dropout's
keep-count-per-step math, does not apply here since all three stay well above the 30-dim
override threshold.

#### Endpoint families

Mean R-squared pooled across the 15 flavors under the frozen protocol, with the stock baseline
and the best single flavor for comparison. Families are assigned by endpoint and dataset name
(the assignment is editorial, not something the analysis code computes); the count in
parentheses is how many of the 32 endpoint-columns fall in each.

| Family | Pooled flavors | Stock | Best flavor |
|---|---|---|---|
| PXR (1) | 0.640 | 0.668 | minimol (0.688) |
| Lipophilicity (3) | 0.536 | 0.562 | surrogate_adme (0.782) |
| Permeability/binding (6) | 0.443 | 0.465 | surrogate_adme (0.558) |
| Potency (2) | 0.411 | 0.490 | jazzy (0.558) |
| Solubility (3) | 0.230 | 0.247 | surrogate_adme (0.334) |
| Clearance (11) | 0.185 | 0.164 | surrogate_adme (0.258) |
| hERG (1) | 0.159 | 0.148 | minimol (0.220) |
| CYP inhibition (5) | 0.122 | 0.118 | surrogate_adme (0.180) |

The family difficulty ordering is stable and unsurprising: PXR and lipophilicity are easy for
everything, CYP inhibition and hERG are hard for everything, which tracks assay noise and
mechanistic directness rather than anything about pretraining.

What the fit-to-purpose premise predicted, and this table does not show, is a family where some
flavor's descriptor block gives it a decisive edge. The pooled flavor mean beats stock on only
three of eight families (clearance, hERG, CYP inhibition), and those three are exactly the
hardest ones, where every column is close to the noise floor. On the five families with real
signal, the average flavor is worse than stock and only the best flavor clears it. Read
conservatively: descriptor pretraining buys a per-endpoint best-of, not a family-level
specialization, and picking the right flavor per endpoint matters more than any flavor's family
profile. Note also that `surrogate_adme` is the best flavor in five of eight families, and it is
the different-corpus arm, so even that best-of read leans on the column that is not directly
comparable.

Per-flavor detail and the 250K-era comparison are below.

### Multi-seed stock baseline and per-cell error bars (2026-07-13)

The report cards averaged each flavor over five finetune seeds but measured them against a
single-seed stock-CheMeleon baseline (seed 42), which made both the R-squared baseline column
and the entire MAE %-change card an unpaired, high-variance reference. The stock reference is
now finetuned at five seeds (frozen: the original seed 42 plus seeds 1-4), and `report_card.py`
averages it. The R-squared card annotates every endpoint cell (flavors and the baseline column)
with a plus/minus seed standard deviation. The MAE %-change card colors a cell only where the
flavor's MAE differs significantly from the baseline's; a non-significant cell is painted white
and every cell is annotated with its p-value, so the card highlights only the differences the
seed spread supports.

That test was originally an uncorrected two-sample Welch t-test run independently in each cell,
which is now corrected; see Multiple comparisons below. Under the corrected test, 95 of 480
cells are significant on the frozen card, on 20 of 32 endpoints, concentrated on the
high-signal, low-variance endpoints (LogD leads with 13 of 15 flavors separating) where flavors
pull away from stock cleanly.

The correction matters most where the single seed happened to be unrepresentative. On PXR:

| baseline | R-squared | MAE |
|---|---|---|
| seed 42 alone (old) | 0.729 | 0.490 |
| 5-seed average (new) | 0.668 +/- 0.084 | 0.550 +/- 0.054 |

Seed 42 was the best of the five stock seeds on PXR (the five range 0.525 to 0.729 in
R-squared), so the old single-seed baseline flattered stock and made every flavor look 9 to 27
percent worse on MAE. Against the 5-seed average, PXR is roughly a wash: minimol (-3.1 percent
MAE versus baseline), rdkit2d (-1.7 percent), and osmordred (-0.2 percent) beat it, and the rest
are worse but not by much. Under the significance test, every flavor on PXR is white, so no flavor
differs significantly from stock on PXR MAE at five seeds; that held under the original
uncorrected test (p from 0.057 to 0.99) and holds more comfortably under the corrected one
(every PXR p at or near 1.0). The earlier "PXR
delta is bad across the board" read was an artifact of comparing 5-seed flavor averages against
one lucky stock point, not a real deficit of descriptor pretraining on PXR. The lesson
generalizes: the stock baseline is as seed-unstable as the flavors, so a single-seed reference is
not a safe comparison for any endpoint with this much finetune-seed variance.

The reduced and unlocked stock baselines are finetuned at seeds 1-5, matching the flavor legs
(seeds 1-4 first, since no single-seed run pre-existed for those protocols, then seed 5 on
2026-07-14). Both flavor legs have now finished, and all four report cards are rendered as of
2026-07-15 (job 1967405); see `TODO.md` Milestone 8.

**Unlocked finetuning buys nothing over frozen; reduced is the protocol that pays.** Averaged
over the 285 flavor-endpoint cells of `plots/lr_ranking_r2.csv` (15 flavors x 19 endpoint names,
each cell a mean over seeds 1-5 and, for the six endpoint names measured in more than one dataset,
over those datasets):

| protocol | mean R-squared | mean delta vs frozen | cells better than frozen |
|---|---|---|---|
| frozen | 0.291 | | |
| reduced | 0.321 | +0.031 | 228 / 285 |
| unlocked | 0.292 | +0.002 | 176 / 285 |

Unlocked lands within 0.002 R-squared of frozen and wins barely more cells than a coin flip
would, so paying for a full-network finetune at 1e-3 recovers the frozen result rather than
improving on it. Reduced beats frozen on 80 percent of cells for the same finetune cost.

The per-protocol stock baselines tell the same story from the other side. These per-flavor means
are taken over all 160 rows a flavor has in `results/lr_metrics.csv` (32 dataset-endpoint cells x
5 seeds), so they weight repeated endpoints by dataset and are not on the collapsed basis of the
table above. Against the unlocked
stock baseline (mean R-squared 0.337), only `surrogate_adme` (0.368) clears it by a meaningful
margin and `rdkit2d` (0.324), the best same-corpus flavor, does not clear it at all. Against the
reduced stock baseline (0.316), four flavors clear it: `surrogate_adme` (0.374), `minimol`
(0.371), `rdkit2d` (0.356), and `osmordred_pca95` (0.345). `surrogate_adme` topping both lists is
the different-corpus reference arm, not an apples-to-apples comparison, so `minimol` and `rdkit2d`
under reduced are the real headline. Descriptor pretraining shows up under reduced and washes out
under unlocked, which is what a protocol that overwrites the pretrained backbone should do.

### Milestone 6 (partial, 250K, superseded by the full-corpus table above)

**Milestone 6 (partial): 9 direct-compute flavors, 250K corpus, `order_fix` prescaling, 24
finetune recipes / 32 endpoint columns (multi-target recipes split into one column per
endpoint), all 3 finetune protocols.** Does not yet include osmordred (pretrained
separately under the milestone-4/5 triage, on the full corpus with `chemeleon_baseline`
prescaling, so not directly comparable on this table) or the milestone-7 learned-model
flavors (not started). Numbers are mean R-squared from `results/metrics.csv` (frozen) and
`results/lr_metrics.csv` (reduced/unlocked); full detail in `plots/report_card_r2.csv`.

Mean R-squared by flavor, frozen protocol (the sweep's primary protocol):

| Flavor | Frozen | Reduced | Unlocked |
|---|---|---|---|
| rdkit2d | 0.350 | 0.371 | 0.309 |
| jazzy | 0.314 | 0.332 | 0.333 |
| pubchem | 0.285 | 0.327 | 0.295 |
| usrcat | 0.280 | 0.310 | 0.316 |
| erg | 0.273 | 0.336 | 0.317 |
| ecfp | 0.259 | 0.277 | 0.268 |
| atompair | 0.259 | 0.299 | 0.301 |
| whim | 0.221 | 0.282 | 0.324 |
| e3fp | 0.211 | 0.244 | 0.265 |

`rdkit2d` wins frozen and reduced by a clear margin (16 of 32 endpoint-columns, out of 32,
have `rdkit2d` as the single best flavor). Under unlocked, the ranking compresses and
reorders: `jazzy` edges out `whim` (0.333 vs. 0.324) while `rdkit2d` drops to 4th (0.309),
the same "unlocked compresses the spread, prescaling/target matters less once the
backbone can move" pattern seen in the prescaling LR sweep. `reduced` is again the best
protocol on average across flavors (mean 0.309 vs. 0.272 frozen, 0.303 unlocked), also
matching the prescaling sweep's finding.

Endpoint-family read (mean R-squared pooled across all 9 flavors, family assigned by
recipe/endpoint name keyword):

| Family | Mean R-squared | Best flavor |
|---|---|---|
| PXR | 0.660 | rdkit2d (0.701) |
| Lipophilicity (LogD) | 0.446 | rdkit2d (0.656) |
| Permeability/binding (MDR1, Caco-2, plasma-protein binding) | 0.423 | rdkit2d (0.495) |
| Potency (pIC50) | 0.389 | usrcat (0.602) |
| Solubility | 0.216 | rdkit2d (0.300) |
| Clearance | 0.185 | rdkit2d (0.246) |
| hERG | 0.162 | usrcat (0.223) |
| CYP inhibition | 0.112 | pubchem (0.157) |

PXR and lipophilicity are the easiest endpoints for every flavor; hERG and CYP inhibition
are the hardest for every flavor, consistent with those being noisier, more
mechanistically indirect assays regardless of pretraining target. `rdkit2d` is the best
or near-best flavor on every family except potency and hERG, where `usrcat` (a 3D
shape/pharmacophore descriptor) wins instead, the one specialization signal in this
partial sweep: 3D shape appears to carry more signal for potency and hERG (both driven by
binding-site geometry) than for the ADMET properties `rdkit2d` otherwise dominates.

## Multiple comparisons

**The MAE %-change card ran one uncorrected test per cell, and now corrects within each endpoint
row.** Every cell asked "does this flavor's MAE differ from stock on this endpoint" with an
independent two-sample Welch t-test, 480 times per card. With 15 flavors measured against one
shared control, that is 15 comparisons per row, so the card's false-positive count grew with the
number of flavors shown rather than staying at the nominal 5%.

The fix is Dunnett's test, which is built for this design: many treatments against one control,
family-wise error controlled across the family, and a variance pooled across all groups in the
family. One endpoint row is one family. Correcting per row rather than across the whole card is
a deliberate choice, since each endpoint is a separate question; error across the card's 32 rows
is therefore not controlled, and a reader scanning the whole card for the single greenest cell
is still exposed to that.

What it changes, per protocol, on cells colored at p at or below 0.05:

| protocol | uncorrected (old) | Dunnett (current) | lost | newly significant |
|---|---|---|---|---|
| frozen | 143 | 95 | 51 | 3 |
| reduced | 126 | 106 | 30 | 10 |
| unlocked | 67 | 40 | 30 | 3 |

Roughly a third of the frozen card's colored cells did not survive. Nothing in the qualitative
read changed: the flavors that separate most (`surrogate_adme` on 13 endpoints, `whim`, `ecfp`
and `e3fp` on 9-10 as clear losses) and the endpoints that separate most (LogD, clearance) are
the same ones, and PXR remains entirely white.

**Some cells got *more* significant, which is worth understanding rather than glossing.** In
7-18% of cells the corrected p-value is smaller than the uncorrected one. Dunnett pools variance
across the whole family, so a flavor whose own seed spread is much wider than the pool borrows
precision from its neighbours, and the resulting gain in error degrees of freedom (about 64,
against the 5 to 8 a two-group Welch test scrapes together at five seeds) can outweigh the
multiplicity penalty. That is a symptom of unequal variances between flavors, not a bug, and it
cuts both ways: quiet flavors are penalized by the same pooling.

**The assumption this rests on is not verifiable at five seeds.** Dunnett assumes the groups
share a common variance. Within an endpoint row the observed variance ratios run a median of 31x
and a maximum of 352x, which is large; Levene rejects on almost no rows, but Levene has very
little power with 16 groups of 5, so that non-rejection is not evidence of homogeneity. The
honest statement is that pooling is an assumption made for the degrees of freedom it buys, and
that the direction of its error differs per flavor. A variance-free alternative (Holm on the
Welch p-values, valid under arbitrary dependence) is more conservative still: 74 cells on the
frozen card against Dunnett's 95.

**The AVERAGE row is gated on its own test.** The row summarizes each flavor across all 32
endpoints, which is a different question from any cell above it, so it runs its own Dunnett
family rather than inheriting the per-cell p-values: one value per finetune seed, that seed's
mean MAE %-change across the card's endpoints, against the baseline seeds put through the same
aggregation. It used to be colored by its mean change unconditionally, which read as a verdict
the seed spread often did not support. Colored AVERAGE cells now stand at 7 of 15 flavors under
frozen (3 better, 4 worse), 7 of 15 under reduced (4 better, 3 worse), and 4 of 15 under
unlocked (1 better, 3 worse). The change bites hardest where a mean looked meaningful and was
not: 7 frozen columns, 4 reduced and 10 unlocked, carry a mean change of at least one percentage
point and are still painted white. Under frozen, `osmordred` at -2% and `jazzy` at -2% are two
such columns.

Two properties of that row follow from the pooling and are worth holding in mind. The verdict
runs on the mean shift against the family's pooled spread rather than the column's own, so a
single noisy column raises the bar for every other column, and two columns with equal mean shifts
get equal p-values however different their own spreads. And the control group is centered at
exactly zero by construction, since it is the baseline measured against its own per-endpoint
means; its spread, not its location, is what the treatments are judged against.

**What no correction fixes.** The five seeds are finetune seeds off a single seed-42 pretraining
run per flavor. Every p-value on every card therefore speaks to finetune variance only, and none
of them licenses a claim about the pretraining target having produced a better foundation, since
pretraining variance is unsampled at n=1. This is the larger threat to the study's central claim
than the multiplicity was.

## Standalone controls

Both studies below are standalone by construction: they write their own metrics CSV, never the
shared `results/metrics.csv`, and no column of theirs appears on the report card.

### osmordred_surrogate: was surrogate_adme's lead its corpus or its target?

`surrogate_adme` leads every protocol, but it is confounded two ways against sweep `osmordred`:
a different corpus (Novartis molecules) and a different target (25 ADME predictions against
3585 osmordred descriptors). This control computes the osmordred descriptor target on
`surrogate_adme`'s Novartis corpus, holding the target identical to sweep `osmordred` and the
corpus identical to `surrogate_adme`, so where it lands attributes the lead. Frozen protocol, 5
finetune seeds off one s42 foundation; `results/osmordred_surrogate_metrics.csv`.

| condition | frozen mean R-squared |
|---|---|
| surrogate_adme (Novartis corpus, ADME target) | 0.369 +/- 0.010 |
| **osmordred_surrogate (Novartis corpus, osmordred target)** | **0.325 +/- 0.004** |
| sweep osmordred (shared corpus, osmordred target) | 0.305 +/- 0.016 |
| chemeleon_stock | 0.295 +/- 0.009 |

The control is 5 seeds; the three reference arms are 6 (the legacy seed 42 plus 1-5), which is
why their values differ by ~0.001-0.004 from the same flavors on the report card above.

The control lands nearest sweep osmordred (|delta| 0.020) rather than surrogate_adme (0.044).
Holding the Novartis corpus while swapping the ADME target for the descriptor target drops
transfer from 0.369 to 0.325, so **surrogate_adme's lead came mostly from its on-task ADME
target, not its chemical space.** The space does contribute something: the control clears stock
by +0.031 (Welch t=7.06, p<0.001) and sits above sweep osmordred, so Novartis molecules are a
modestly better pretraining set than the shared PubChem corpus for these endpoints. But the
target dominates.

The practical implication is uncomfortable for the fit-to-purpose premise: the strongest column
on the card is strong because its pretraining target is itself a set of ADME predictions, which
is closer to distilling an existing ADME model than to learning a general representation. The
Novartis corpus is ~273K molecules against the sweep's 944K, a size confound that runs against
this reading rather than for it (a smaller corpus scoring higher strengthens the case that
corpus is not what mattered).

### PXR external test: the internal specialization does not transfer

The sweep's PXR endpoint splits `pxr_pec50.parquet` with an inline Butina `ClusterSplitter`, so
split membership moves with the finetune seed. This rerun instead evaluates every flavor plus
stock on two fixed external hold-outs, the OpenADMET PXR-challenge Phase 1 (253 molecules) and
Phase 2 (260), with a single fixed 90/10 train/val split (1950/217, seed 42) shared across every
flavor and seed, so the finetune seed varies only head init and training. Reduced protocol, 5
seeds; `results/pxr_ext_metrics.csv`.

Sorted by phase 1; an asterisk marks a delta against stock that is significant at Welch
p at or below 0.05.

| flavor | Phase 1 R-squared | Phase 2 R-squared |
|---|---|---|
| surrogate_adme | 0.361 +/- 0.027 | 0.415 +/- 0.028 |
| erg | 0.336 +/- 0.032 | 0.348 +/- 0.015* |
| **chemeleon_stock** | **0.325 +/- 0.043** | **0.413 +/- 0.021** |
| minimol | 0.299 +/- 0.036 | 0.372 +/- 0.027* |
| jazzy | 0.287 +/- 0.038 | 0.321 +/- 0.055* |
| atompair | 0.280 +/- 0.044 | 0.385 +/- 0.033 |
| usrcat | 0.245 +/- 0.026* | 0.334 +/- 0.046* |
| osmordred | 0.241 +/- 0.054* | 0.331 +/- 0.054* |
| rdkit2d | 0.234 +/- 0.078 | 0.336 +/- 0.024* |
| osmordred_pca95 | 0.216 +/- 0.028* | 0.333 +/- 0.033* |
| osmordred_pca80 | 0.204 +/- 0.059* | 0.359 +/- 0.024* |
| whim | 0.157 +/- 0.041* | 0.230 +/- 0.040* |
| osmordred_pca90 | 0.134 +/- 0.033* | 0.300 +/- 0.013* |
| pubchem | 0.134 +/- 0.047* | 0.258 +/- 0.030* |
| ecfp | 0.036 +/- 0.043* | 0.100 +/- 0.045* |
| e3fp | 0.027 +/- 0.024* | 0.065 +/- 0.035* |

**No pretrained flavor significantly beats stock CheMeleon on either phase.** `surrogate_adme`
is the only flavor above stock on both, and neither margin is significant (+0.036 phase 1,
+0.003 phase 2). Most flavors land significantly below.

The load-bearing result is `rdkit2d`. On the sweep's internal Butina-split PXR column it was the
leading flavor, which was the cleanest specialization signal the whole study produced. Here it
sits mid-pack and below stock on both phases (0.234, 0.336). **A specialization measured on an
internally generated split did not survive a fixed external hold-out**, which is a warning about
how much weight any single report-card cell can carry, not just about PXR. Phase 2 is easier
than phase 1 for every model (stock 0.413 against 0.325), so the two phases differ in difficulty
as well as membership.

The binary fingerprints fail here far worse than on the report card (phase 1 `ecfp` 0.036,
`e3fp` 0.027 against stock 0.325), consistent with their leaky-pretext read but much starker on
molecules that are genuinely held out.

## External foundations: pretraining corpus and size

A comparison of pretraining datasets rather than descriptor blocks. Four externally pretrained
CheMeleon-format foundations, carrying no target or pretraining in this repo, were finetuned on
the same 24 endpoints at 5 seeds under all three protocols (1440 finetunes): `molpile_1M`,
`molpile_5M`, `molpile_10M`, and `expansion_gen`. Each checkpoint passed the same
`{hyper_parameters, state_dict}` format and message-passing-dim validation gate the repo's own
foundations do. The existing 5-seed stock-CheMeleon baseline is the reference column.
`results/external_metrics.csv` and `plots/external_foundations/`; the sweep's
`results/metrics.csv` is untouched.

Mean R-squared across the 32 endpoint-columns, per seed then averaged over seeds. The frozen
stock column here is 6 seeds (the legacy seed 42 plus 1-5) rather than the report card's 5
(42 plus 1-4), because `results/external_metrics.csv` folded in the seed-5 stock run; that
accounts for the 0.001 difference between this row and the report card's, and the reduced and
unlocked stock columns are the same 5 seeds in both.

| foundation | frozen | reduced | unlocked |
|---|---|---|---|
| **chemeleon_stock** | **0.295 +/- 0.009** | **0.316 +/- 0.014** | **0.337 +/- 0.008** |
| molpile_5M | 0.255 +/- 0.009 | 0.264 +/- 0.015 | 0.242 +/- 0.025 |
| molpile_10M | 0.220 +/- 0.013 | 0.250 +/- 0.009 | 0.213 +/- 0.007 |
| molpile_1M | 0.217 +/- 0.015 | 0.241 +/- 0.022 | 0.292 +/- 0.019 |
| expansion_gen | 0.186 +/- 0.013 | 0.250 +/- 0.016 | 0.231 +/- 0.015 |

**Every external foundation loses to stock CheMeleon under every protocol, and all twelve
deficits are significant** (Welch against the same-protocol stock seeds, worst p 0.003; largest
deficits are `expansion_gen` frozen at -0.109 and `molpile_10M` unlocked at -0.124).

**Pretraining size does not buy accuracy monotonically.** `molpile_5M` leads `1M` and `10M`
under frozen and reduced, but under unlocked the order inverts to `1M` first and `10M` last. The
spread between the three sizes (0.04-0.08) is the same scale as the deficit to stock, and the
ordering is not stable across protocols, so a 10x corpus increase is not visible as a
consistent gain.

Read this as a weaker comparison than the flavor sweep, and say so plainly: the four checkpoints
differ in corpus, size, and pretraining recipe simultaneously, so a per-foundation delta cannot
be attributed to corpus size alone. What it does support is a negative claim that matters for
the study's framing: stock CheMeleon is a strong baseline that is not easily beaten by
pretraining on more molecules, which is consistent with the flavor sweep's finding that beating
it takes a well-chosen target rather than more data.

## Per-flavor read

The reads below were written against the 250K partial sweep and are kept for their per-flavor
reasoning; the current numbers are the 5-seed table under Report card, not the values quoted
here. Where the two disagree, the 5-seed table wins. The notable revisions: `rdkit2d` remains
the strongest same-corpus direct-compute descriptor but is no longer the top flavor overall
(`minimol` is, among same-corpus flavors), and its PXR specialization does not survive the
external test above; `usrcat`'s potency and hERG wins do not reappear on the full-corpus 5-seed
card, where `jazzy` takes potency and `minimol` takes hERG.

- **rdkit2d**: strongest general foundation in this partial sweep, winning 16 of 32
  endpoint-columns frozen and leading 6 of 8 endpoint families. Confirms the "continuous
  descriptor flavors are strong general foundations" prior, at least relative to the other
  8 flavors tested so far (osmordred not yet in this comparison).
- **jazzy**: second-strongest frozen/reduced, and the unlocked-protocol winner. This result
  predates the target-dropout invariant: it was pretrained under the fixed 0.85 masked-pretext
  dropout (`jazzy` is only 6 dims, so under 1 target/step on average) yet did not underperform.
  Under the full-corpus rerun, jazzy and every other sub-threshold flavor pretrain at
  `dropout_fraction=0.0` per the now-enforced invariant, so re-read this number as a
  pre-invariant data point, not the current regime.
- **usrcat**: mid-table on mean R-squared but wins potency and hERG specifically, both
  binding-driven endpoints. The one specialization result in this sweep: a 3D
  shape/pharmacophore descriptor target transfers better than 2D descriptors for
  endpoints that hinge on molecular shape and binding-site fit.
- **ecfp, atompair, pubchem, e3fp** (binary fingerprint flavors): bottom half of the
  ranking on mean R-squared (pubchem highest of the four at 0.285, e3fp lowest overall at
  0.211), consistent with the "leaky/weak pretext" prior in the methodology watch-items.
  Not uniformly worst, though: pubchem wins the CYP family, and atompair and e3fp each win
  one endpoint-column outright.
- **erg, whim**: unremarkable frozen/reduced, but both climb under unlocked (erg to 0.317,
  whim to 0.324, whim's frozen-to-unlocked jump the largest in the table), suggesting
  their pretext targets need more backbone adaptation to pay off than rdkit2d's does.
- **osmordred**: not in this table (pretrained and finetuned separately under the
  milestone-4/5 triage, full corpus, `chemeleon_baseline` prescaling); its frozen mean
  R-squared there was 0.352, in the same range as `rdkit2d`'s 0.350 here, but the two runs
  differ in corpus size and prescaling recipe, so this is context, not a controlled
  comparison. A controlled osmordred-vs-Milestone-6 comparison needs osmordred rerun under
  the Milestone-6 protocol (250K corpus, `order_fix`).
- **minimol**: Milestone 7, finetune complete (frozen protocol, 250K corpus). Mean
  R-squared 0.360 across 32 endpoint-columns, the highest single-flavor mean seen so far
  in this sweep (edges `rdkit2d`'s 0.350), though not yet compared to Milestone 6 in one
  merged table. A learned 512-dim embedding transferring at least as well as the strongest
  direct-compute descriptor is a positive early signal for the learned-flavor family,
  worth confirming once `surrogate_adme` lands and a merged comparison is run.
- **surrogate_adme**: fanning out in the full-corpus rerun. Was held out of the
  250K Milestone 7 pending the target-dropout-fraction question (a 25-dim target, where
  the regime-default 0.85 keeps under one target per step). That question is settled: it is
  sub-threshold, so it pretrains at `dropout_fraction=0.0` under the hard invariant enforced
  in `train.py` (`config.DROPOUT_OVERRIDE_MAX_DIM=30`), with no per-flavor decision left to
  make and no nonzero override permitted. `TODO.md`'s Future experiments entry has the full
  account, including the enforcement path (`losses.py`, `train.py`, `config.py`).

## Meta-model

**Current (frozen protocol, all 15 flavors, 5 seeds).** Stacking with LGBM on out-of-fold
per-flavor predictions beats the best single flavor on 21 of 32 endpoint-columns: mean
R-squared 0.500 for the meta-model against 0.417 for the best single flavor per endpoint
(mean delta +0.082). The stacker is scored independently per seed and averaged, so its column
carries the same seed error bars every report-card cell does; 18 of the 21 wins exceed one
standard deviation of their own delta. `results/meta_model_lgbm.csv` holds the per-endpoint
table.

The flavor the meta-model most often has to beat is `surrogate_adme` (the single-flavor winner
on 18 of 32 endpoint-columns), with `minimol` next at 4. That concentration is itself a caution:
the bar the ensemble is measured against is set largely by the different-corpus reference arm.

The gains are largest where no single flavor does well, and the losses are where one already
does. Biggest wins: Caco-2 efflux (meta 0.710 against `osmordred_pca95` 0.332), ChEMBL RLM
clearance (0.376 against `surrogate_adme` 0.096), Caco-2 permeability (0.667 against
`osmordred` 0.421). Biggest losses: Biogen solubility (0.106 against `surrogate_adme` 0.210)
and Biogen RLM clearance (0.541 against 0.606). Stacking recovers signal that is spread thinly
across flavors and dilutes signal that is concentrated in one.

Not run: the RF and MLP alternative estimators, and the per-mode (reduced/unlocked)
meta-models. The latter are fully wired (`meta_model.py --lr-mode`, writing a mode-scoped
`meta_model_<estimator>_<mode>.csv`) and are the only remaining step from the original plan, so
the ensemble question is currently answered under frozen finetuning only. Given that reduced is
the protocol where pretraining pays for single flavors, the reduced meta-model is the one most
worth running.
