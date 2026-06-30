# SLURM orchestration

Job-array scripts that fan the pipeline across cluster nodes: one task per flavor for target
computation and pretraining, one task per (flavor, endpoint) for finetuning.

## One-shot run

```bash
export SURROGATE_CSV=/path/to/protacdb2.0_zinc_chembl_dataset.csv
bash slurm/run_all.sh
```

`run_all.sh` generates the per-flavor finetuning configs, then submits all five stages as a
SLURM dependency chain (corpus → targets → pretrain → finetune → analyze). Each stage waits
for every task of the previous stage to succeed before starting. Come back when the analyze
job finishes; results land in `results/` and `analysis/plots/`.

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
`analysis/plots/prescaling_ranking_r2.csv` to pick the production recipe.

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
| `pretrain.sbatch` | 13 (GPU array) | targets |
| `finetune.sbatch` | 312 (GPU array) | pretrain |
| `analyze.sbatch` | 1 (GPU) | finetune |

Prescaling triage (driven by `run_ablations.sh`):

| Script | Tasks | Depends on |
|---|---|---|
| `ablation_target.sbatch` | 1 (CPU) | corpus |
| `ablation_prescale.sbatch` | 7 (CPU array) | target |
| `ablation_pretrain.sbatch` | 7 (GPU array) | prescale |
| `ablation_finetune.sbatch` | 168 (GPU array) | pretrain |
| `ablation_analyze.sbatch` | 1 (GPU) | finetune |

The array ranges default to the current registry. If the flavor set changes, update `--array`
in the array scripts and recount recipes:

```bash
# flavor count
conda run -n sarizard python -c \
    "from pretraining.flavors import flavor_names; print(len(flavor_names()))"
# recipe count (after configs.generate)
ls configs/*/*.yaml | grep -v /_baseline/ | wc -l
# ablation count (sets ablation_prescale / ablation_pretrain arrays)
conda run -n sarizard python -c \
    "from pretraining.prescaling import ablation_names; print(len(ablation_names()))"
# ablation recipe count (sets ablation_finetune array)
ls configs/ablation_*/*.yaml | wc -l
```

## Notes

- Every stage is resumable: a flavor or recipe whose output already exists is skipped, so a
  re-submission only fills gaps.
- The learned-model flavors (minimol, ml_qm) need their isolated envs created before the
  targets stage (`conda env create -f envs/minimol.yml`, `conda env create -f envs/mlqm.yml`).
  `surrogate_adme` runs in the main env and only needs the released CSV.
- Logs land in `slurm/logs/` (gitignored).
- The committed `configs/_baseline/` recipes are stock CheMeleon; the finetune array excludes
  them. Run them separately if you want a stock-CheMeleon reference column.
