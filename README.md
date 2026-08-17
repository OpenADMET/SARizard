# SARizard

![banner](assets/banner.jpg)

Structure-Activity Relationship Wizard. A rapid-turnaround study of whether the
descriptor target a molecular foundation model is pretrained against determines
which downstream endpoints it serves best.

CheMeleon is a directed message-passing neural network (D-MPNN, Chemprop) pretrained
self-supervised to regress a block of precomputed molecular descriptors from the
molecular graph. SARizard trains several "flavors" of that foundation, each varying
only the descriptor block it learns to predict (osmordred and PCA-compressed variants of it,
RDKit descriptors, fingerprints, pharmacophores, 3D descriptors, and learned-model targets
such as minimol embeddings and surrogate ADME predictions). Every flavor is pretrained
separately on one shared corpus, never concatenated, then finetuned on real ADMET
benchmark endpoints and evaluated. The result is a fit-to-purpose catalog: which
foundational pretraining target lends itself to which endpoint and endpoint family.

## Experiment design

1. **One shared corpus.** `corpus/corpus_full.parquet`: 944,296 molecules, the whole
   cleaned 1M-molecule PubChem set from the CheMeleon Training Data record that survives
   salt stripping and canonicalization. Every flavor computes its target on these same
   molecules so the report-card columns are apples-to-apples. A 250K subset
   (`corpus/corpus_250k.parquet`) was built first and used only for early screening; every
   result on record was rerun on the full corpus, and the 250K flavor-sweep artifacts are
   archived under `archive/flavor_sweep_250k/`. The 250K subset is still what the pipeline
   builds by default, so full-corpus runs export
   `CORPUS_FILE=corpus/corpus_full.parquet CORPUS_N=1000000` (see [Running the full
   experiment](#running-the-full-experiment)).
2. **Pretrain per flavor.** Each flavor pretrains a D-MPNN to regress its descriptor
   block under one fixed regime (shared corpus, capped epochs, shared LR schedule, gradient
   clipping, masked-pretext target dropout). Continuous targets are prescaled with the
   `chemeleon_baseline` recipe the ablation triage picked and train with MSE; binary
   fingerprint targets skip scaling and train with BCE. Target blocks at or under 30 dims
   (`jazzy` at 6, `surrogate_adme` at 25) pretrain with the masked-pretext dropout off, which
   is a hard invariant rather than a tunable (see `AGENTS.md`).
3. **Convert.** The pretrained message-passing block is exported to the checkpoint
   format `openadmet-models` consumes via `from_foundation`.
4. **Finetune per endpoint.** For every flavor, each ADMET benchmark endpoint is
   finetuned with `openadmet anvil`, initialized from that flavor's foundation. 24
   finetuning recipes across 8 source datasets become 32 endpoint-columns on the card. Each
   flavor is pretrained once at seed 42 and finetuned at 5 replicate seeds, under three
   backbone protocols: `frozen` (`mpnn_lr = 0`), `reduced` (`mpnn_lr = ffn_lr/10`), and
   `unlocked` (`mpnn_lr = ffn_lr`). Each protocol is compared against a stock-CheMeleon
   baseline finetuned the same way; the `reduced` baseline was later deepened to 20 seeds,
   which is the allocation Dunnett's test wants.
5. **Evaluate and compare.** Three artifacts. A pair of report-card heatmaps (endpoints by
   flavors): the R-squared card leads with the stock-CheMeleon baseline column, divided from
   the flavor block by a heavy rule, and the MAE %-change card colors each flavor's change
   against that baseline, white where Dunnett's test does not separate the two. Each card is
   written as a PNG and as an equivalent HTML table. Endpoint rows are bracketed by source
   dataset, and the three ChEMBL-derived sources are named for the assay they measure (ChEMBL
   CLint, ChEMBL CYP, and, for the single hERG row, ChEMBL) rather than for a release number.
   A pair of summary boxplots then unpacks each card's AVERAGE row into one box per column
   (endpoint-to-endpoint spread as the box, seed spread of the average as the error bar),
   ordered best to worst left to right. Finally a meta-model stacks the per-flavor predictions
   per endpoint to test whether an ensemble of foundations beats the best single one.

## Flavors

| Flavor | Pretraining target | dim | Loss | Target source |
|---|---|---|---|---|
| osmordred | osmordred descriptors | 3585 | MSE | direct compute (isolated env) |
| rdkit2d | RDKit 2D descriptors | 200 | MSE | direct compute |
| erg | ERG pharmacophore | 315 | MSE | direct compute |
| ecfp | ECFP bits | 2048 | BCE | direct compute |
| atompair | Atom-pair bits | 2048 | BCE | direct compute |
| pubchem | PubChem keys | 881 | BCE | direct compute |
| usrcat | USRCAT (3D) | 60 | MSE | direct compute (conformers) |
| whim | WHIM (3D) | 114 | MSE | direct compute (conformers) |
| e3fp | E3FP bits (3D) | 1024 | BCE | direct compute (conformers) |
| jazzy | hydration energy + H-bond strengths | 6 | MSE | direct compute (isolated env) |
| minimol | minimol embedding | 512 | MSE | learned model (isolated env) |
| surrogate_adme | 25 ADME predictions | 25 | MSE | released dataset (native corpus) |
| osmordred_pca80 | osmordred, PCA to 80% explained variance | 70 | MSE | derived from osmordred |
| osmordred_pca90 | osmordred, PCA to 90% explained variance | 147 | MSE | derived from osmordred |
| osmordred_pca95 | osmordred, PCA to 95% explained variance | 237 | MSE | derived from osmordred |

Binary fingerprint targets are deterministic from the input graph, so they are a
leaky and likely weak pretext; their position on the report card is itself a result.
The 3D flavors require conformer generation (RDKit ETKDG + MMFF94) and their targets are
not bit-reproducible across runs (conformers are stochastic). The learned-model flavors generate their target by running the source model over the shared
corpus. The three PCA flavors have no calculator of their own: they compress the
`full`-recipe-prescaled `osmordred` matrix, with the PCA fit on the train rows only, so they
test whether the 3585-dim block's width or its content is what the foundation is learning.
`surrogate_adme` is the exception to the shared corpus: the Novartis work ships a 273K-row
prediction CSV (CC BY 4.0) with 25 ADME values per molecule, so this flavor uses those
molecules and labels directly as its own pretraining corpus rather than predicting over the
shared corpus (see `sarizard/pretraining/features/surrogate_target.py`). Its report-card
column is therefore a different-corpus reference arm rather than an apples-to-apples
comparison.

Each calculator writes a plain `cache/targets/<flavor>/target.npy` in the flavor's own
environment, then a pack step in the main environment converts it to the chunked
`cache/targets/<flavor>/target.zarr` the trainer reads. This keeps zarr (and its Python
3.11+ requirement) out of the old, conflicting learned-model environments.

## What it found

`FINDINGS.md` is the source of record; `wiki/` holds the per-flavor and per-endpoint detail.
The sweep on record covers all 15 flavors under all three finetune protocols at 5 finetune
seeds each, against a stock-CheMeleon baseline per protocol (20 seeds under `reduced`), plus
the prescaling triage, the external-foundation comparison, the `osmordred_surrogate`
chemical-space control, and the PXR external-test rerun. In short:

- **The descriptor target matters, but less than the finetune protocol.** Under frozen and
  reduced, the learned-model flavors (`surrogate_adme`, `minimol`) and `rdkit2d` clear the
  stock baseline significantly and the binary fingerprints fall well below it. Under unlocked,
  almost everything collapses back to or below stock, so the pretraining target shows through
  only while the backbone is held still or nearly still.
- **Specialization is real but narrow.** The per-endpoint winner changes across families, but
  no flavor beats stock across a whole family the way the fit-to-purpose premise predicts, and
  the one clean specialization signal (`rdkit2d` on PXR) does not survive a fixed external
  hold-out.
- **Stacking beats any single foundation.** The LGBM meta-model wins 21 of 32
  endpoint-columns, mean delta R-squared +0.082 over the best single flavor per endpoint. This
  is the frozen protocol; the per-protocol meta-models are wired but were never launched.
- **Pretraining corpus size is not the lever.** Four external foundations spanning 1M to 10M
  molecules all lose to stock CheMeleon under every protocol, significantly, and the three
  Molpile sizes do not order monotonically.
- **Control depth changed the reading.** Deepening the reduced-protocol stock baseline from 5
  seeds to 20 moved seven flavors rather than three above stock, so part of what read as "the
  target barely matters" was a thin baseline rather than the foundations.

## Layout

All Python code lives in the importable `sarizard/` package; the repo root holds inputs,
recipes, and regenerable artifacts.

- `sarizard/` is the package (`pip install -e .`); run modules from the repo root with
  `python -m sarizard.<sub>.<module>`:
  - `sarizard/corpus/` prepares the shared SMILES corpus from the original CheMeleon corpus.
  - `sarizard/pretraining/` vendors and adapts `how-to-train-your-chemeleon`: per-flavor target
    calculators in `features/`, the pretrain config, `prescaling.py` (the toggleable descriptor
    preprocessing and its ablation registry), and `convert_checkpoint.py`.
  - `sarizard/configs/` is the per-flavor recipe generator.
  - `sarizard/analysis/` holds the report cards, the AVERAGE-row summary boxplots, the
    prescaling and LR reports, the side-study reports, and the meta-model;
    `sarizard/analysis/paths.py` is the single source of truth for on-disk locations.
- `configs/_baseline/` holds the committed stock-CheMeleon recipe templates; generated per-flavor
  recipes land in `configs/<flavor>__s<seed>/` (gitignored).
- `corpus/`, `data/` hold inputs; `cache/`, `foundations/`, `results/`, `plots/` hold regenerable
  artifacts (all gitignored; `data/` is committed, see `data/README.md`).
- `slurm/` holds the sbatch job-array scripts for parallel pretraining and finetuning, plus the
  `run_*.sh` drivers that submit them; shared runtime configuration lives in `slurm/env.sh`.
- `archive/` holds runs moved aside rather than overwritten: the 250K flavor sweep, the
  invalid pre-gradient-clipping ablation triage, and the full-corpus ablation artifacts parked
  there so the 250K corpus-size check could reuse the same paths.
- `wiki/` is an Obsidian vault with detailed per-flavor and per-endpoint write-ups and figures.

## Setup

The project targets Python 3.11+ and uses conda for environments (see `AGENTS.md`).

The fastest path on a cluster is `setup.sh`, which builds every environment from `envs/*.yml`,
installs SARizard into each (with `--no-deps` so the isolated envs' pinned stacks are left
intact), runs the test suite in the main env, and prints `okay` if everything passed:

```bash
bash setup.sh                 # all envs, then test
bash setup.sh main osmordred  # restrict to specific envs (envs/<name>.yml basenames)
FORCE=1 bash setup.sh         # recreate envs that already exist
SKIP_OSMORDRED_BUILD=1 bash setup.sh   # skip the slow osmordred source build
```

To set up only the main environment by hand:

```
conda env create -f envs/main.yml
conda activate sarizard
pip install -e ../openadmet-models
pip install -e .
```

`envs/main.yml` deliberately does not pin a PyTorch build, since the right one depends on the
node's CUDA. Install it after creating the env, or the conda-forge CPU-only build gets picked up
and pretraining silently runs on CPU:

```bash
conda install -n sarizard pytorch pytorch-cuda=<ver> -c pytorch -c nvidia
```

Three target generators with conflicting stacks have isolated environments
(`envs/osmordred.yml`, `envs/jazzy.yml`, `envs/minimol.yml`). Each writes a plain `target.npy` to `cache/` that the main
environment packs to zarr, so their dependencies never reach this training env. osmordred has
no package release and is built from source into its env with `envs/build_osmordred.sh`;
minimol needs the PyG extension wheels declared in its env file's find-links.
`surrogate_adme` runs in the main environment; it reads the Novartis released CSV directly
and requires no isolated env.

Finetuning and analysis run in a separate `openadmet` environment (`envs/openadmet.yml`, built
by `setup.sh`), not the main env: they drive the openadmet-models CLI, whose stack (pandas 2.x,
torch 2.7, Python 3.12) conflicts with the main training env (pandas 3.x, torch 2.12, Python
3.11). Point `OPENADMET_ENV` at it (default `openadmet`).

## Running the full experiment

Build the environments first (`bash setup.sh`, see [Setup](#setup)), then:

```bash
# surrogate_adme needs the Novartis released CSV; the pipeline defaults to
# cache/surrogate/protacdb2.0_zinc_chembl_dataset.csv (see wiki/surrogate_adme.md to download).
# Only export SURROGATE_CSV to point somewhere else:
# export SURROGATE_CSV=/elsewhere/protacdb2.0_zinc_chembl_dataset.csv

# the full corpus every recorded result ran on; slurm/env.sh still defaults to the 250K
# screening subset, so export both to reproduce the sweep as run
export CORPUS_FILE=corpus/corpus_full.parquet CORPUS_N=1000000

# submit the full pipeline and walk away
bash slurm/run_all.sh

# how the recorded sweep was actually run: one foundation per flavor pretrained at seed 42,
# then five finetune replicates off it (this path skips corpus/target/split/pretrain)
FOUNDATION_SEED=42 FLAVOR_SEEDS="1 2 3 4 5" bash slurm/run_all.sh
```

`run_all.sh` generates the per-(flavor, seed) finetuning configs, then runs six stages in order
(corpus preparation, target computation, split, pretraining, finetuning, analysis). The
pre-finetune stages go out as a SLURM dependency chain; the finetune stage then runs in batches
through `submit_batched.sh`, which submits a batch, waits, reruns any casualties, and only then
submits the next, so the driver blocks for hours and wants a persistent shell (an interactive
allocation or `nohup`). Results land in `results/` and `plots/` when the final job finishes.
Every stage is resumable: existing outputs are skipped, so re-running with more seeds fills in
only the new ones. `FLAVOR_SEEDS` alone pretrains a separate foundation per seed, tagged
`<flavor>__s<seed>`; with `FOUNDATION_SEED` set, the seeds become finetune replicates off that
one foundation, which is how the sweep on record was run. The report card and meta-model average
the seeds back to one column per flavor. See `slurm/README.md` for the per-stage scripts and how
to resubmit after a partial failure.

After the sweep, `bash slurm/run_lr_experiments.sh` repeats the finetuning from the same
foundations with the MPNN backbone partially unfrozen (`reduced`) or fully unfrozen
(`unlocked`) and compares both against the frozen sweep in `plots/lr_ranking_r2.csv`.

`slurm/run_stock_baseline.sh` finetunes the released stock-CheMeleon checkpoint on the same
endpoints under the same recipes; that is the baseline column every card is measured against, so
it has to exist before the analysis is meaningful.

Three further drivers cover the side studies, all on the same endpoints:
`slurm/run_external_foundations.sh` finetunes four foreign CheMeleon-format checkpoints
(`molpile_1M`, `molpile_5M`, `molpile_10M`, `expansion_gen`), varying the pretraining dataset and
its size instead of the descriptor block; `slurm/run_pxr_ext.sh` retests the PXR specialization
signal against the two OpenADMET PXR-challenge phases as fixed external hold-outs, instead of the
sweep's internal Butina split; and `slurm/run_osmordred_surrogate.sh` computes the osmordred
target on `surrogate_adme`'s corpus, separating that flavor's chemical space from its target.

### Prescaling ablation triage (run first)

Before the flavor sweep, decide how continuous descriptor targets are preprocessed. The
triage drives one representative flavor (osmordred) through every prescaling recipe in
`sarizard/pretraining/prescaling.py` with the backbone, corpus, and regime held fixed, then compares
downstream endpoint performance so the difference is the prescaling alone:

```bash
conda env create -f envs/osmordred.yml   # the triage flavor's target environment
conda activate sarizard-osmordred && bash envs/build_osmordred.sh && conda deactivate
bash slurm/run_ablations.sh
```

Read `plots/prescaling_ranking_r2.csv` and the ablation report card to pick the production
recipe, then bake it into the core workflow before running the flavor sweep. **Outcome:**
`chemeleon_baseline` won under the frozen protocol on the full corpus, narrowly over
`order_fix`. That recipe is what `split.py` already produces, so nothing needed baking in. The
same triage on the 250K corpus ranks the recipes differently, which is itself a result: corpus
size shapes which prescaling wins, so a screening-scale triage does not transfer.

### Regenerating the figures

The analysis modules run from the repo root against the tidy metrics CSVs, so a figure can be
re-rendered without resubmitting anything. `sarizard.analysis.evaluate` needs the `openadmet`
env (it runs predictions) and the 600 dpi report cards need more memory than a login shell
allows, so both go through `slurm/analyze.sbatch`. The summary boxplots read only the metrics
CSV and depend on nothing beyond pandas, numpy, and matplotlib, so they run inline in either
env. For example, the reduced-protocol AVERAGE-row summaries for the three card sets:

```bash
python -m sarizard.analysis.average_summary --lr-mode reduced \
    --metrics-csv results/lr_metrics.csv --baseline-flavor chemeleon_stock_reduced \
    --exclude-recipe cyp1a2_st expansionrx_logd_st_rand
python -m sarizard.analysis.average_summary --lr-mode reduced \
    --metrics-csv results/ablation_metrics.csv --baseline-flavor chemeleon_stock_reduced \
    --ablations --prefix ablation_average_summary \
    --exclude-recipe cyp1a2_st expansionrx_logd_st_rand
python -m sarizard.analysis.average_summary --lr-mode reduced \
    --metrics-csv results/external_metrics.csv --baseline-flavor chemeleon_stock_reduced \
    --columns molpile_1M molpile_5M molpile_10M expansion_gen \
    --out-dir plots/external_foundations \
    --exclude-recipe cyp1a2_st expansionrx_logd_st_rand
```

## Reproducing a single flavor

The scikit-fingerprints flavors (`ecfp`, `rdkit2d`, `erg`, `atompair`, `pubchem`, `usrcat`,
`whim`, `e3fp`) and `surrogate_adme` compute in the main `sarizard` env; only `osmordred`,
`jazzy`, and `minimol` have isolated envs. Compute the target in the flavor's env
(`get_flavor(name).env`), pack and train in the main env. Example with `ecfp` (main env
throughout) at the default seed 42:

```bash
conda activate sarizard
# --corpus is explicit: the flag defaults to the 250K screening subset, not the full corpus
python -m sarizard.pretraining.features.compute_target --flavor ecfp \
    --corpus corpus/corpus_full.parquet
python -m sarizard.pretraining.features.pack_target --flavor ecfp

# split (seed-independent), then pretrain the foundation at seed 42. The SMILES parquet must be
# the same one the target was computed on, or split.py fails on the row-count mismatch
cd sarizard/pretraining
python split.py --input-zarr ../../cache/targets/ecfp/target.zarr \
    --input-smiles ../../corpus/corpus_full.parquet --outdir ../../cache/splits/ecfp \
    --flavor ecfp --force
python train.py --flavor ecfp --input-dir ../../cache/splits/ecfp \
    --output-dir runs/ecfp__s42 --foundation-name ecfp__s42_mp.pt --seed 42
cd ../..

# generate the seeded recipes and finetune one endpoint
python -m sarizard.configs.generate --flavors ecfp
openadmet anvil --recipe-path configs/ecfp__s42/cyp_mt.yaml --output-dir results/ecfp__s42/cyp_mt/
```

An isolated flavor differs only in the compute step: `conda activate sarizard-osmordred`
(after `bash envs/build_osmordred.sh`) for `compute_target --flavor osmordred`, then switch
back to `sarizard` for pack, split, train, and finetune.

Datasets and the pretraining corpus are not redistributed; regenerate or obtain them
and place them as described before training.
