---
tags: [method, status/blue]
---
# Shared Corpus and Regime

> **Summary:** The one thing held constant across every flavor. A fixed 250K-molecule subset
> of the original CheMeleon PubChem corpus, a single seed, and one pretraining regime (capped
> epochs, shared LR schedule, mean aggregation, default graph featurizer). The only intended
> difference between flavors is the target block (and MSE vs BCE for binary targets), so the
> [[Report Card]] columns are comparable.

## What is fixed

- Corpus: `corpus/corpus_250k.parquet`, downsampled once from the Zenodo PubChem set (seed 42).
- Featurizer: chemprop DEFAULT (`MultiHotAtomFeaturizer.v2` + `MultiHotBondFeaturizer`), so
  `d_v`/`d_e` match openadmet's finetuning featurizer.
- Aggregation: mean (openadmet rebuilds foundations with mean aggregation).
- Regime: epoch cap, patience, LR schedule, and model width all in `sarizard/pretraining/config.py`.
- Storage chunking is fixed across flavors, so the train/val split selects the same molecules
  for every flavor and the pretraining batch size is identical.
- Descriptor prescaling: one recipe, chosen once by the [[Prescaling Ablation]] triage and
  applied identically to every continuous flavor.

## Seeds

The regime pins a single training seed by default, so each flavor is one foundation. To
separate a flavor's effect from initialization noise, the sweep can pretrain it at several
seeds (`FLAVOR_SEEDS`), tagged `<flavor>__s<seed>`; the [[Report Card]] and [[Meta-Model]]
average the seeds back per flavor. The seed varies only pretraining (the finetune seed is
held fixed), so the spread it reveals is foundation-initialization variance. See
[[Finetune Protocols]].

## Why it matters

A fit-to-purpose comparison only means something if the foundations differ in exactly one
way. Any drift in corpus, featurizer, aggregation, or regime would confound the report card.

## Related

- Center of the flavor graph; every flavor branches off this node.
- Compared against the published [[Stock CheMeleon]] (different corpus and regime).
