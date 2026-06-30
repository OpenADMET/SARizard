# Finetuning recipes

`anvil` recipes for finetuning each foundation flavor on every ADMET benchmark endpoint.

## Layout

- `_baseline/` holds the 24 stock-CheMeleon recipes (one per dataset and endpoint), copied
  from the sibling igm project. They are the source templates and the only committed recipes.
- `generate.py` reads those templates and writes `configs/<flavor>/<endpoint>.yaml` for every
  flavor, changing only the foundation the model initializes from and the run labels. The
  generated per-flavor directories are gitignored; regenerate them with the generator.

## Generate

```
python -m configs.generate                       # all flavors, all endpoints
python -m configs.generate --flavors ecfp jazzy  # a subset
```

Each generated recipe sets `procedure.model.params.from_foundation` to
`foundations/<flavor>_mp.pt` (resolved when `anvil` runs from the repo root), normalizes the
lightning `accelerator` to `auto`, and relabels `metadata.name`/`tag`/`tags` from the
backbone to the flavor. Everything else is inherited from the baseline, so the finetuning
regime is identical across flavors and only the foundation differs.

## Stock-CheMeleon reference

The `_baseline/` recipes keep `from_foundation: chemeleon`, so running them directly gives a
stock-CheMeleon reference column. It uses a different pretraining corpus and regime than our
flavors, so treat it as an external reference, not an apples-to-apples arm of the study.
