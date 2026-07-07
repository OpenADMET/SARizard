# Project Context — SARizard

SARizard trains "flavors" of the CheMeleon molecular foundation model that differ
only in the descriptor block each is pretrained to regress, then finetunes and
evaluates each flavor on ADMET benchmark endpoints to build a fit-to-purpose catalog.
The narrative goal and headline results live in `FINDINGS.md`; open avenues in `TODO.md`.

## Operating model

- Environments are managed with conda, not uv. All environment manifests live in `envs/`.
  The main environment is `envs/main.yml`; the conflicting target generators have isolated
  environments under the same directory (`envs/osmordred.yml`, `envs/jazzy.yml` for its exact
  RDKit pin, `envs/minimol.yml`, `envs/mlqm.yml`). osmordred has no package release and is
  built from source into its env via `envs/build_osmordred.sh` after the env is created;
  minimol pulls the PyG C++ extensions from the find-links pinned in its manifest. Declare a
  dependency in the relevant manifest before installing; never `pip install` or `conda install`
  ad hoc into a shared env.
- The analysis package runs from the repo root as `python -m sarizard.analysis.<module>`.
  `sarizard/analysis/paths.py` is the single source of truth for on-disk locations; scripts
  never hardcode experiment paths.
- Prefer small, incremental, reviewable changes; follow the commit conventions in the
  global instructions (bare imperative subject, no conventional-commit prefixes, no
  authorship footers).
- **`TODO.md` is the source of truth for what has run, not the filesystem.** Update it in
  the same session as any action that changes shared state: submitting a job that
  computes/recomputes a target, changes the corpus a milestone runs against, archives
  results, or otherwise moves a milestone forward. Write the update immediately after
  submitting or confirming completion, not deferred to "later" or the next session; a gap
  between what ran and what `TODO.md` says ran is the failure mode to avoid; do not require
  the next reader to reconstruct it from `sacct`/`squeue` history and file mtimes. Do not
  leave code that implements a recorded decision (e.g. an override, a new flavor, a
  pipeline change) uncommitted across a session boundary; commit it as soon as it is
  tested, even if the run it enables hasn't finished yet. If a scope decision changes
  mid-flight (e.g. which corpus a milestone runs against), edit the milestone's own
  entry to say so explicitly rather than adding a note only at the point of change; a
  reader jumping to the milestone should not see stale scope.

## Repository layout

All Python code lives in the importable `sarizard/` package (`pip install -e .`); the repo
root holds inputs, recipes, and regenerable artifacts. Run modules from the repo root as
`python -m sarizard.<sub>.<module>`.

- `sarizard/corpus/` shared corpus preparation (full corpus by default as of the
  full-corpus rerun; still parameterized by size for screening runs); `sarizard/pretraining/` vendored and
  adapted `how-to-train-your-chemeleon` (`features/` holds one target calculator per flavor,
  `prescaling.py` the toggleable descriptor preprocessing and ablation registry,
  `convert_checkpoint.py` exports foundations); `sarizard/configs/` the recipe generator;
  `sarizard/analysis/` the report card and meta-model.
- `configs/_baseline/` committed stock-CheMeleon templates; generated `configs/<flavor>__s<seed>/`,
  `configs/ablation_<name>__s<seed>/`, and `configs/lr_<mode>__<flavor>__s<seed>/` recipes are gitignored.
- `corpus/`, `data/` inputs; `cache/`, `foundations/`, `results/`, `plots/`,
  `sarizard/pretraining/runs/` regenerable artifacts (gitignored; `data/` is committed,
  provenance in `data/README.md`).
- `slurm/` sbatch job-array scripts: `run_all.sh` drives the flavor sweep, `run_ablations.sh`
  the prescaling triage that precedes it. `wiki/` Obsidian vault; `tests/` the test suite.

## Compatibility invariants (do not break silently)

These keep a self-trained foundation loadable by `openadmet-models` and prevent a
train/serve representation mismatch. A violation does not error loudly, it degrades the
foundation, so treat each as a gate.

- **Aggregation.** `openadmet-models` hardcodes `MeanAggregation` when loading a
  foundation (`openadmet/models/architecture/chemprop.py`). Pretraining must use
  `MeanAggregation`, not the upstream how-to-train default of `NormAggregation`.
- **Graph featurizer.** The message-passing input dims (`d_v`, `d_e`) are baked into the
  checkpoint, so the pretraining graph featurizer must match openadmet's
  `ChemPropFeaturizer`. Pretrain with the `DEFAULT` featurizer
  (`MultiHotAtomFeaturizer.v2` + `MultiHotBondFeaturizer`), not `RIGR`. Verify the dims
  align before fanning out.
- **Checkpoint format.** `openadmet` expects `torch.load(path, weights_only=True)` to
  return `{"hyper_parameters": <BondMessagePassing kwargs>, "state_dict": <mp state>}`.
  `convert_checkpoint.py` produces exactly this from the pretrained MPNN.

## Experiment discipline

- **One shared corpus.** Every flavor computes its target on the same fixed corpus
  (single seed, persisted). Do not let a flavor drift to a different molecule set; the
  report-card columns must be comparable. As of the full-corpus rerun, that shared corpus
  is `corpus/corpus_full.parquet` (944,296 molecules, `CORPUS_FILE`/`CORPUS_N` in
  `slurm/env.sh`), superseding the 250K screening corpus (`corpus/corpus_250k.parquet`)
  the sweep originally ran on; the 250K results are archived at `archive/flavor_sweep_250k/`.
  Learned-model flavors run their source model over this same corpus. The lone exception is `surrogate_adme`: its
  target (the 25 Novartis ADME predictions) is only defined on the Novartis molecules, so
  that flavor pretrains on its own corpus (`compute_target --flavor surrogate_adme` writes
  `corpus_smiles.parquet` alongside its `target.npy`, and `split.py` uses it). Its column is
  therefore a different-corpus reference arm, not an apples-to-apples comparison; keep it
  labelled as such. No other flavor may use a non-shared corpus.
- **One fixed pretraining regime.** Corpus size, epoch cap, and LR schedule are shared
  across flavors and recorded in each flavor's pretrain config; the only intended
  difference between flavors is the target block (and MSE vs BCE for binary targets).
- **Cache targets once, in two steps.** Each flavor's calculator runs in its own
  environment and writes a plain `cache/targets/<flavor>/target.npy` (numpy only), so the
  conflicting learned-model environments never need zarr. `sarizard.pretraining.features.pack_target`
  then converts that `.npy` into the chunked `target.zarr` the trainer reads, in the main
  environment. Targets are expensive and deterministic (3D conformer targets excepted);
  compute each once and reuse. Storage chunk rows are fixed across flavors
  (`config.CORPUS_CHUNK_ROWS`) so the train/val split and pretraining batch size are
  identical for every flavor.
- **No leakage.** Per-flavor target scaling (winsorize and z-score for continuous
  targets) is fit on the pretraining train split only. `prescaling.py` fits every step
  (percentiles, correlation, Yeo-Johnson lambdas, variance, mean/std) on the train chunks
  from the shared chunk split (`sarizard/pretraining/splitting.py`), so prescaling and `split.py`
  hold out the same molecules. Binary targets skip scaling and train with BCE. The
  meta-model trains on out-of-fold predictions and evaluates on the held-out test split; it
  never sees in-sample finetuned predictions.
- **Prescaling is chosen once, then fixed.** `prescaling.py` makes each preprocessing step
  (NaN/inf clean, winsorize, correlated-column drop, Yeo-Johnson, low-variance drop,
  z-score) toggleable so the milestone-4 ablation triage (`slurm/run_ablations.sh`, one
  flavor, fixed backbone) can pick a recipe. Once picked, bake it into the core workflow and
  apply it identically to every continuous flavor; varying prescaling mid-sweep confounds the
  report card like varying the backbone would.

## Standing gates

- ruff (lint + format) and pyright (basic) run via pre-commit on the package; the
  vendored `sarizard/pretraining/` tree relaxes docstring and import-order gates to stay close to
  upstream. notebooks are exempt from docstring and import-order gates but not correctness.
- Commit-message conventions are enforced by a commit-msg hook.

## Reference repositories (read-only, on this machine)

- `../information-gain-metric` the sibling project this is modeled on (data, baseline
  recipes, the report-card heatmap in `analysis/analyze.py`).
- `../openadmet-models` the training and inference library (`anvil` recipes,
  `ChemPropModel.from_foundation`, `load_anvil_model_and_metadata`).
- `../how-to-train-your-chemeleon` the upstream pretraining pipeline vendored here.
