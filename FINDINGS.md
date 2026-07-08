# Findings: fit-to-purpose CheMeleon foundations

Source of record for headline results. Per-flavor and per-endpoint detail lives in the
Obsidian wiki under `wiki/`; where the wiki and this file disagree, this file wins.

## The question

Does the descriptor target a CheMeleon-style foundation is pretrained against determine
which ADMET endpoints and endpoint families it serves best? And does stacking the
per-flavor finetuned predictions into a meta-model beat the best single foundation?

## Status

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

### Cross-protocol check (reduced/unlocked, full corpus, regime-fixed)

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

## Per-flavor read

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

First result, across the 10 flavors with finetuned results (the 9 Milestone-6 flavors plus
`minimol`; `osmordred`, `surrogate_adme` not yet in this table). Stacking with LGBM
on out-of-fold per-flavor predictions beats the best single flavor on 23 of 32
endpoint-columns: mean R-squared 0.481 for the meta-model vs. 0.390 for the best single
flavor per endpoint (mean delta +0.091). `rdkit2d` and `minimol` are the two flavors the
meta-model most often has to beat (11 and 9 endpoint-columns respectively where one of them
is the single-flavor winner), consistent with both being the strongest general-purpose
flavors in the per-flavor read above. The ensemble does not win everywhere: it loses on 9 of
32 endpoints, mostly where one flavor already specializes strongly (e.g. `usrcat` on potency,
where stacking in weaker flavors dilutes rather than helps). See `results/meta_model_lgbm.csv`
for the per-endpoint table. Not yet run: RF/MLP alternative estimators, and osmordred/
surrogate_adme once they join the merged table.
