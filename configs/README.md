# Finetuning recipes

`anvil` recipes for finetuning each foundation flavor on every ADMET benchmark endpoint.

## Layout

- `_baseline/` holds the 24 stock-CheMeleon recipes (one per dataset and endpoint), copied
  from the sibling igm project. They are the source templates and the only committed recipes.
- `generate.py` reads those templates and writes `configs/<flavor>__s<seed>/<endpoint>.yaml`
  for every flavor and seed, changing only the foundation the model initializes from and the
  run labels. The generated directories are gitignored; regenerate them with the generator.

## Generate

```
python -m sarizard.configs.generate                        # all flavors, seed 42
python -m sarizard.configs.generate --seeds 1 2 3          # one recipe set per (flavor, seed)
python -m sarizard.configs.generate --flavors ecfp jazzy   # a subset of flavors
```

Each generated recipe sets `procedure.model.params.from_foundation` to
`foundations/<flavor>__s<seed>_mp.pt` (resolved when `anvil` runs from the repo root), sets the
backbone learning rate by protocol (`--mpnn-lr-mode`, default `frozen` = `mpnn_lr: 0`, so
finetuning measures representation quality rather than initialization luck), normalizes the
lightning `accelerator` to `auto`, and relabels `metadata.name`/`tag`/`tags` from the backbone
to the variant. Everything else is inherited from the baseline, so the finetuning regime is
identical across flavors and only the foundation (and, in the LR experiments, `mpnn_lr`) differs.

## Learning-rate experiments

`--mpnn-lr-mode` and `--label-prefix` reuse the flavor foundations to sweep the backbone
finetune protocol (`reduced` = `mpnn_lr = ffn_lr/10`, `unlocked` = `mpnn_lr = ffn_lr`), writing
to a namespaced directory. `slurm/run_lr_experiments.sh` calls this once per mode:

```
python -m sarizard.configs.generate --seeds 1 2 3 \
    --mpnn-lr-mode reduced --label-prefix lr_reduced      # -> configs/lr_reduced__<flavor>__s<seed>/
```

## Prescaling ablation recipes

The prescaling triage (see `sarizard/pretraining/prescaling.py` and `slurm/run_ablations.sh`) reuses
the same templates against ablation foundations. The ablation mode points one explicit
foundation at a named output directory:

```
python -m sarizard.configs.generate \
    --foundation foundations/ablation_order_fix__s42_mp.pt \
    --out-subdir ablation_order_fix__s42
```

This writes `configs/<out-subdir>/<endpoint>.yaml` for every endpoint, labeled with the
out-subdir. `run_ablations.sh` calls this once per (ablation, seed), so the directories it
generates are `configs/ablation_<name>__s<seed>/`; they are gitignored like the per-flavor ones.

## Stock-CheMeleon reference

The `_baseline/` recipes keep `from_foundation: chemeleon`, so running them directly gives a
stock-CheMeleon reference column. It uses a different pretraining corpus and regime than our
flavors, so treat it as an external reference, not an apples-to-apples arm of the study.
