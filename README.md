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
5. **Evaluate and compare.** Two artifacts: a report-card heatmap (endpoints by
   flavors, one selectable metric) and a meta-model that stacks the per-flavor
   predictions per endpoint to test whether an ensemble of foundations beats the
   best single one.

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
| ml_qm | qmdesc QM descriptors (pooled) | 24 | MSE | learned model (isolated env) |

Binary fingerprint targets are deterministic from the input graph, so they are a
leaky and likely weak pretext; their position on the report card is itself a result.
The 3D flavors require conformer generation (RDKit ETKDG + MMFF94) and their targets are
not bit-reproducible across runs (conformers are stochastic). The learned-model flavors generate their target by running the source model over the shared
corpus. `surrogate_adme` is the exception: the Novartis work ships a 273K-row prediction CSV
(CC BY 4.0) with 25 ADME values per molecule, so this flavor uses those molecules and labels
directly as its own pretraining corpus rather than predicting over the shared 250K set (see
`pretraining/features/surrogate_target.py`).

Each calculator writes a plain `cache/targets/<flavor>/target.npy` in the flavor's own
environment, then a pack step in the main environment converts it to the chunked
`cache/targets/<flavor>/target.zarr` the trainer reads. This keeps zarr (and its Python
3.11+ requirement) out of the old, conflicting learned-model environments.

## Layout

- `corpus/` prepares the shared 250K SMILES subset from the original CheMeleon corpus.
- `cache/` holds per-flavor descriptor targets and rescaled stores (gitignored, computed once).
- `pretraining/` vendors and adapts `how-to-train-your-chemeleon`: per-flavor target
  calculators in `features/`, per-flavor pretrain configs, and `convert_checkpoint.py`.
- `foundations/` holds the converted foundation checkpoints (gitignored).
- `configs/<flavor>/` holds one finetuning recipe per endpoint, generated from the baseline recipes.
- `slurm/` holds the sbatch job-array scripts for parallel pretraining and finetuning.
- `analysis/` is the importable analysis package; run modules from the repo root with
  `python -m analysis.<module>`. `analysis/paths.py` is the single source of on-disk locations.
- `data/` holds the benchmark sets and splits (gitignored; see `data/README.md`).
- `wiki/` is an Obsidian vault with detailed per-flavor and per-endpoint write-ups and figures.

## Setup

The project targets Python 3.11+ and uses conda for environments (see `AGENTS.md`).

```
conda env create -f envs/main.yml
conda activate sarizard
pip install -e ../openadmet-models
pip install -e .
```

Four target generators with conflicting stacks have isolated environments
(`envs/osmordred.yml`, `envs/jazzy.yml`, `envs/minimol.yml`, `envs/mlqm.yml`). Each writes a plain `target.npy` to `cache/` that the main
environment packs to zarr, so their dependencies never reach this training env.
`surrogate_adme` runs in the main environment; it reads the Novartis released CSV directly
and requires no isolated env.

## Running the full experiment

Create the conda environments, then:

```bash
# one-time: create isolated envs for the four conflicting target generators
conda env create -f envs/osmordred.yml
conda env create -f envs/jazzy.yml
conda env create -f envs/minimol.yml
conda env create -f envs/mlqm.yml

# set the path to the Novartis surrogate-ADME CSV (download link in surrogate_target.py)
export SURROGATE_CSV=/path/to/protacdb2.0_zinc_chembl_dataset.csv

# submit the full pipeline as a SLURM dependency chain and walk away
bash slurm/run_all.sh
```

`run_all.sh` generates the per-flavor finetuning configs, then submits five stages in order
(corpus preparation, target computation, pretraining, finetuning, analysis), each gated by
the previous stage completing without errors. Results land in `results/` and `analysis/plots/`
when the final job finishes. See `slurm/README.md` for the per-stage scripts and how to
resubmit after a partial failure.

## Reproducing a single flavor

```bash
# compute target in the flavor's environment, pack in the main env
conda activate sarizard-ecfp      # or sarizard-osmordred / -jazzy / -minimol / -mlqm
python -m pretraining.features.compute_target --flavor ecfp
conda activate sarizard
python -m pretraining.features.pack_target --flavor ecfp

# pretrain, then finetune one endpoint
sbatch slurm/pretrain.sbatch      # runs only the flavor if foundation already exists for others
python -m configs.generate --flavors ecfp
openadmet anvil --recipe-path configs/ecfp/cyp_mt.yaml --output-dir results/ecfp/cyp_mt/
```

Datasets and the pretraining corpus are not redistributed; regenerate or obtain them
and place them as described before training.
