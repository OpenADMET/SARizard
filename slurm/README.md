# SLURM orchestration

Job-array scripts that fan the pipeline across cluster nodes: one task per flavor for target
computation and pretraining, one task per (flavor, endpoint) for finetuning.

## Before submitting

Edit the resource directives at the top of each `.sbatch` file (the `EDIT_PARTITION` and
`EDIT_ACCOUNT` placeholders, plus time, GPU, CPU, and memory) to match your cluster. Runtime
settings (the main env name, accelerator, repo path) live in `env.sh` and can be exported
instead of edited. Create the conda environments first (`envs/main.yml` and the isolated envs in `envs/`) and
prepare the shared corpus (`python -m corpus.prepare_corpus`). Submit every job from the repo root so `SLURM_SUBMIT_DIR` resolves.

The array ranges default to the current registry (13 flavors, 312 recipes). If the flavor set
changes, update `--array`:

```
# flavors (compute_targets, pretrain)
conda run -n sarizard python -c "from pretraining.flavors import flavor_names as f; print(len(f()))"
# recipes (finetune), after generating them
python -m configs.generate && ls configs/*/*.yaml | grep -v /_baseline/ | wc -l
```

## Pipeline

```
sbatch slurm/compute_targets.sbatch     # per flavor: compute target (flavor env) + pack (main env)
sbatch slurm/pretrain.sbatch            # per flavor: split + pretrain + export foundation (GPU)
python -m configs.generate              # write per-flavor finetuning recipes
sbatch slurm/finetune.sbatch            # per (flavor, endpoint): anvil finetune (GPU)
python -m analysis.evaluate --accelerator gpu   # collect test metrics + cache predictions
python -m analysis.report_card --metric r2      # report-card heatmap
python -m analysis.meta_model                   # stacking vs best single flavor
```

Chain stages with dependencies if you want one submission to gate the next, for example:

```
tid=$(sbatch --parsable slurm/compute_targets.sbatch)
sbatch --dependency=afterok:$tid slurm/pretrain.sbatch
```

## Notes

- Every stage is resumable: a flavor or recipe whose output already exists is skipped, so a
  re-submission only fills gaps.
- The learned-model flavors (minimol, ml_qm) need their isolated envs created before running
  their array tasks (`conda env create -f envs/minimol.yml`, `conda env create -f envs/mlqm.yml`).
  `surrogate_adme` runs in the main env and only needs the released CSV (see
  `pretraining/features/surrogate_target.py`).
- Logs land in `slurm/logs/` (gitignored).
- The committed `configs/_baseline/` recipes are stock CheMeleon; the finetune array excludes
  them. Run them separately if you want a stock-CheMeleon reference column.
```
