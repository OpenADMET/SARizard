# SARizard

![banner](assets/banner.jpg)

Structure-Activity Relationship Wizard. A rapid-turnaround study of whether the
descriptor target a molecular foundation model is pretrained against determines
which downstream endpoints it serves best.

CheMeleon is a directed message-passing neural network (D-MPNN, Chemprop) pretrained
self-supervised to regress a block of precomputed molecular descriptors from the
molecular graph. SARizard trains several "flavors" of that foundation, each varying
only the descriptor block it learns to predict (osmordred, RDKit descriptors,
fingerprints, pharmacophores, 3D descriptors, and learned-model targets such as
minimol embeddings and surrogate ADME predictions). Every flavor is pretrained
separately on one shared corpus, never concatenated, then finetuned on real ADMET
benchmark endpoints and evaluated. The result is a fit-to-purpose catalog: which
foundational pretraining target lends itself to which endpoint and endpoint family.

## Experiment design

1. **One shared corpus.** A fixed 250K-molecule subset of the original CheMeleon
   PubChem corpus (downsampled once, single seed). Every flavor computes its target
   on these same molecules so the report-card columns are apples-to-apples.
2. **Pretrain per flavor.** Each flavor pretrains a D-MPNN to regress its descriptor
   block under one fixed reduced regime (shared corpus, capped epochs, shared LR
   schedule). Continuous targets train with MSE, binary fingerprint targets with BCE.
3. **Convert.** The pretrained message-passing block is exported to the checkpoint
   format `openadmet-models` consumes via `from_foundation`.
4. **Finetune per endpoint.** For every flavor, each ADMET benchmark endpoint is
   finetuned with `openadmet anvil`, initialized from that flavor's foundation.
5. **Evaluate and compare.** Two artifacts: a pair of report-card heatmaps (endpoints by
   flavors) and a meta-model that stacks the per-flavor predictions per endpoint to test
   whether an ensemble of foundations beats the best single one. The R-squared card leads
   with the stock-CheMeleon baseline column, divided from the flavor block by a heavy rule;
   the MAE %-change card colors each flavor's change against that baseline, white where
   Dunnett's test does not separate the two. Each card is written as a PNG and as an
   equivalent HTML table. Endpoint rows are bracketed by source dataset, and the three
   ChEMBL-derived sources are named for the assay they measure (ChEMBL CLint, ChEMBL CYP,
   and, for the single hERG row, ChEMBL) rather than for a release number.

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

Binary fingerprint targets are deterministic from the input graph, so they are a
leaky and likely weak pretext; their position on the report card is itself a result.
The 3D flavors require conformer generation (RDKit ETKDG + MMFF94) and their targets are
not bit-reproducible across runs (conformers are stochastic). The learned-model flavors generate their target by running the source model over the shared
corpus. `surrogate_adme` is the exception: the Novartis work ships a 273K-row prediction CSV
(CC BY 4.0) with 25 ADME values per molecule, so this flavor uses those molecules and labels
directly as its own pretraining corpus rather than predicting over the shared 250K set (see
`sarizard/pretraining/features/surrogate_target.py`).

Each calculator writes a plain `cache/targets/<flavor>/target.npy` in the flavor's own
environment, then a pack step in the main environment converts it to the chunked
`cache/targets/<flavor>/target.zarr` the trainer reads. This keeps zarr (and its Python
3.11+ requirement) out of the old, conflicting learned-model environments.

## Layout

All Python code lives in the importable `sarizard/` package; the repo root holds inputs,
recipes, and regenerable artifacts.

- `sarizard/` is the package (`pip install -e .`); run modules from the repo root with
  `python -m sarizard.<sub>.<module>`:
  - `sarizard/corpus/` prepares the shared 250K SMILES subset from the original CheMeleon corpus.
  - `sarizard/pretraining/` vendors and adapts `how-to-train-your-chemeleon`: per-flavor target
    calculators in `features/`, the pretrain config, `prescaling.py` (the toggleable descriptor
    preprocessing and its ablation registry), and `convert_checkpoint.py`.
  - `sarizard/configs/` is the per-flavor recipe generator.
  - `sarizard/analysis/` is the report card and meta-model; `sarizard/analysis/paths.py` is the
    single source of truth for on-disk locations.
- `configs/_baseline/` holds the committed stock-CheMeleon recipe templates; generated per-flavor
  recipes land in `configs/<flavor>__s<seed>/` (gitignored).
- `corpus/`, `data/` hold inputs; `cache/`, `foundations/`, `results/`, `plots/` hold regenerable
  artifacts (all gitignored; `data/` is committed, see `data/README.md`).
- `slurm/` holds the sbatch job-array scripts for parallel pretraining and finetuning.
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

# submit the full pipeline as a SLURM dependency chain and walk away
bash slurm/run_all.sh

# optional: pretrain each flavor at several seeds to average out initialization noise
FLAVOR_SEEDS="1 2 3" bash slurm/run_all.sh
```

`run_all.sh` generates the per-(flavor, seed) finetuning configs, then submits six stages in
order (corpus preparation, target computation, split, pretraining, finetuning, analysis), each
gated by the previous stage completing without errors. Results land in `results/` and `plots/`
when the final job finishes. `FLAVOR_SEEDS` (default one seed) pretrains each flavor at several
seeds, tagged `<flavor>__s<seed>`; the report card and meta-model average them back per flavor,
and re-running with more seeds fills in only the new ones. See `slurm/README.md` for the
per-stage scripts and how to resubmit after a partial failure.

After the sweep, `bash slurm/run_lr_experiments.sh` repeats the finetuning from the same
foundations with the MPNN backbone partially unfrozen (`reduced`) or fully unfrozen
(`unlocked`) and compares both against the frozen sweep in `plots/lr_ranking_r2.csv`.

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

Read `plots/prescaling_ranking_r2.csv` and the ablation report card to pick the
production recipe, then bake it into the core workflow before running the flavor sweep.

## Reproducing a single flavor

The scikit-fingerprints flavors (`ecfp`, `rdkit2d`, `erg`, `atompair`, `pubchem`, `usrcat`,
`whim`, `e3fp`) and `surrogate_adme` compute in the main `sarizard` env; only `osmordred`,
`jazzy`, and `minimol` have isolated envs. Compute the target in the flavor's env
(`get_flavor(name).env`), pack and train in the main env. Example with `ecfp` (main env
throughout) at the default seed 42:

```bash
conda activate sarizard
python -m sarizard.pretraining.features.compute_target --flavor ecfp
python -m sarizard.pretraining.features.pack_target --flavor ecfp

# split (seed-independent), then pretrain the foundation at seed 42
cd sarizard/pretraining
python split.py --input-zarr ../../cache/targets/ecfp/target.zarr \
    --input-smiles ../../corpus/corpus_250k.parquet --outdir ../../cache/splits/ecfp \
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
