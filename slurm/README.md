# SLURM orchestration

Job-array scripts that fan the pipeline across cluster nodes: one task per flavor for target
computation and pretraining, one task per (flavor, endpoint) for finetuning.

## One-shot run

```bash
# surrogate_adme defaults to cache/surrogate/protacdb2.0_zinc_chembl_dataset.csv;
# export SURROGATE_CSV only if the CSV lives elsewhere
bash slurm/run_all.sh
```

`run_all.sh` generates the per-(flavor, seed) finetuning configs, then submits all six stages
as a SLURM dependency chain (corpus → targets → split → pretrain → finetune → analyze). Each
stage waits for every task of the previous stage to succeed before starting. Come back when the
analyze job finishes; results land in `results/` and `plots/`.

Set `FLAVOR_SEEDS` (default `42`) to pretrain each flavor at several seeds and average out
initialization noise, e.g. `FLAVOR_SEEDS="1 2 3" bash slurm/run_all.sh`. Each seed is its own
foundation/recipes/results (`<flavor>__s<seed>`); the report card and meta-model average the
seeds back to one column per flavor. Re-running with more seeds skips the ones already done and
fills in the rest (see the notes below).

### Prescaling ablation triage (run before the flavor sweep)

```bash
conda env create -f envs/osmordred.yml   # the triage flavor's target environment
conda activate sarizard-osmordred && bash envs/build_osmordred.sh && conda deactivate
bash slurm/run_ablations.sh
```

`run_ablations.sh` drives one representative flavor (`ABLATION_FLAVOR`, default osmordred)
through every prescaling recipe and submits its own chain (corpus → target → prescale →
pretrain → finetune → analyze) with the `ablation_*.sbatch` scripts. The backbone, corpus,
and regime are fixed, so the comparison isolates the prescaling. Read
`plots/prescaling_ranking_r2.csv` to pick the production recipe.

### Finetune LR experiments (run after the flavor sweep)

```bash
bash slurm/run_lr_experiments.sh
```

`run_lr_experiments.sh` reuses the flavor-sweep foundations and repeats the finetuning with the
MPNN backbone partially unfrozen (`reduced`, `mpnn_lr = ffn_lr/10`) or fully unfrozen
(`unlocked`, `mpnn_lr = ffn_lr`), then compares both against the frozen sweep (`lr_finetune` →
`lr_analyze`). Set `LR_MODES` to choose protocols and `FLAVOR_SEEDS` to match the sweep that
produced the foundations. Read `plots/lr_ranking_r2.csv` for each mode's mean R² delta and win
count versus frozen. The frozen-warmup protocol is not included (it needs a two-phase schedule
anvil cannot express; see TODO.md).

## Before submitting

Adjust the time, CPU, and memory directives in each `.sbatch` header if your cluster requires
different limits. Runtime settings live in `env.sh` and can be exported instead of edited
(`REPO_DIR`, `MAIN_ENV`, `ACCELERATOR`). Create conda environments first (`envs/main.yml`
and the isolated envs in `envs/`).

## Scripts

| Script | Tasks | Depends on |
|---|---|---|
| `prepare_corpus.sbatch` | 1 (CPU) | — |
| `compute_targets.sbatch` | 13 (CPU array) | corpus |
| `split.sbatch` | 13 (CPU array) | targets |
| `pretrain.sbatch` | 13 x seeds (GPU array) | split |
| `finetune.sbatch` | 312 x seeds (GPU array) | pretrain |
| `analyze.sbatch` | 1 (GPU) | finetune |

Prescaling triage (driven by `run_ablations.sh`):

| Script | Tasks | Depends on |
|---|---|---|
| `ablation_target.sbatch` | 1 (CPU) | corpus |
| `ablation_prescale.sbatch` | 7 (CPU array) | target |
| `ablation_pretrain.sbatch` | 7 x seeds (GPU array) | prescale |
| `ablation_finetune.sbatch` | 168 x seeds (GPU array) | pretrain |
| `ablation_analyze.sbatch` | 1 (GPU) | finetune |

Finetune LR experiments (driven by `run_lr_experiments.sh`, after the flavor sweep):

| Script | Tasks | Depends on |
|---|---|---|
| `lr_finetune.sbatch` | 312 x seeds x modes (GPU array) | flavor-sweep foundations |
| `lr_analyze.sbatch` | 1 (GPU) | lr_finetune |

`ablation_pretrain` and `ablation_finetune` scale with `ABLATION_SEEDS` (default one seed):
each ablation is pretrained once per seed (`ablation_<name>__s<seed>`), and `ablation_analyze`
averages the seeds back to one column per ablation. `run_ablations.sh` sizes every array
automatically; the counts below are only needed to submit a stage standalone.

```bash
# flavor count (sets the targets/split array)
conda run -n sarizard python -c \
    "from sarizard.pretraining.flavors import flavor_names; print(len(flavor_names()))"
# flavor x seed count (sets the pretrain array)
(source slurm/env.sh; echo $(( $(flavor_list | wc -l) * $(wc -w <<<"$FLAVOR_SEEDS") )))
# recipe count (after configs.generate --seeds); registry flavors x seeds, excludes ablation/lr dirs
(source slurm/env.sh; flavor_recipe_list | wc -l)
# ablation count (sets the ablation_prescale array)
conda run -n sarizard python -c \
    "from sarizard.pretraining.prescaling import ablation_names; print(len(ablation_names()))"
# ablation x seed count (sets the ablation_pretrain array)
(source slurm/env.sh; echo $(( $(ablation_list | wc -l) * $(wc -w <<<"$ABLATION_SEEDS") )))
# ablation recipe count (sets ablation_finetune array)
ls configs/ablation_*/*.yaml | wc -l
```

## Sharding slow target flavors

The 3D conformer flavors (usrcat, whim, e3fp) and jazzy compute far slower than the rest and a
whole-corpus run can exceed the target wall time. Split one across array tasks instead of the
single `compute_targets.sbatch` task:

```bash
NUM_SHARDS=20
SH=$(sbatch --parsable --export=ALL,FLAVOR=usrcat,NUM_SHARDS=$NUM_SHARDS \
    --array=0-$((NUM_SHARDS - 1)) slurm/compute_target_shard.sbatch)
sbatch --export=ALL,FLAVOR=usrcat,NUM_SHARDS=$NUM_SHARDS \
    --dependency=afterok:"$SH" slurm/merge_target.sbatch
```

Each array task computes one contiguous row range into `cache/targets/<flavor>/shards/`. The
merge job concatenates the shards into `target.npy`, refusing to pack if any shard is missing,
mis-shaped, or short of the corpus, then packs the zarr. The fast flavors stay on the default
`compute_targets.sbatch` path.

## Notes

- Every stage is resumable: a flavor or recipe whose output already exists is skipped, so a
  re-submission only fills gaps. This extends to seeds: run with `FLAVOR_SEEDS="1"`, then later
  `FLAVOR_SEEDS="1 2 3"`, and seed 1's foundations and results are skipped while 2 and 3 run.
  Pass the full cumulative seed set (not just the new seeds), since `analyze` averages exactly
  the seeds you name.
- The chain is wired with `afterok`, so a stage is released only if every array task of the
  prior stage succeeds. If one task fails, the dependent stage is cancelled
  (`DependencyNeverSatisfied`) and the chain stops there. Recover by fixing the cause and
  re-running `run_all.sh`: resumability means only the failed gaps recompute and the chain
  re-arms. Read the failing task's `slurm/logs/<stage>_<jobid>_<taskid>.out` first.
- The learned-model flavors (minimol, ml_qm) need their isolated envs created before the
  targets stage (`conda env create -f envs/minimol.yml`, `conda env create -f envs/mlqm.yml`).
  `surrogate_adme` runs in the main env and only needs the released CSV.
- Logs land in `slurm/logs/` (gitignored).
- The committed `configs/_baseline/` recipes are stock CheMeleon; the finetune array excludes
  them. Run them separately if you want a stock-CheMeleon reference column.
