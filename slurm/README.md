# SLURM orchestration

Job-array scripts that fan the pipeline across cluster nodes: one task per flavor for target
computation and pretraining, one task per (flavor, endpoint) for finetuning.

## One-shot run

```bash
# edit partition and account in every .sbatch header first, then:
export SURROGATE_CSV=/path/to/protacdb2.0_zinc_chembl_dataset.csv
bash slurm/run_all.sh
```

`run_all.sh` generates the per-flavor finetuning configs, then submits all five stages as a
SLURM dependency chain (corpus → targets → pretrain → finetune → analyze). Each stage waits
for every task of the previous stage to succeed before starting. Come back when the analyze
job finishes; results land in `results/` and `analysis/plots/`.

## Before submitting

Edit the `EDIT_PARTITION` and `EDIT_ACCOUNT` placeholders (plus time, GPU, CPU, and memory)
in every `.sbatch` header to match your cluster. Runtime settings live in `env.sh` and can be
exported instead of edited (`REPO_DIR`, `MAIN_ENV`, `ACCELERATOR`). Create conda environments
first (`envs/main.yml` and the isolated envs in `envs/`).

## Scripts

| Script | Tasks | Depends on |
|---|---|---|
| `prepare_corpus.sbatch` | 1 (CPU) | — |
| `compute_targets.sbatch` | 13 (CPU array) | corpus |
| `pretrain.sbatch` | 13 (GPU array) | targets |
| `finetune.sbatch` | 312 (GPU array) | pretrain |
| `analyze.sbatch` | 1 (GPU) | finetune |

The array ranges default to the current registry. If the flavor set changes, update `--array`
in the array scripts and recount recipes:

```bash
# flavor count
conda run -n sarizard python -c \
    "from pretraining.flavors import flavor_names; print(len(flavor_names()))"
# recipe count (after configs.generate)
ls configs/*/*.yaml | grep -v /_baseline/ | wc -l
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
