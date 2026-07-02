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
  applied identically to every continuous flavor. **Open caveat:** the triage that picked the
  recipe (`chemeleon_baseline`) ran on the full corpus (944296 molecules), not the 250K
  screening corpus below, and a follow-up check found the prescaling ranking is corpus-size
  sensitive (a different recipe wins on 250K). The recipe was not re-validated on the actual
  250K corpus this regime uses; see [[Prescaling Ablation]] for the numbers.

## Seeds

The regime pins a single training seed by default, so each flavor is one foundation. To
separate a flavor's effect from initialization noise, the sweep can pretrain it at several
seeds (`FLAVOR_SEEDS`), tagged `<flavor>__s<seed>`; the [[Report Card]] and [[Meta-Model]]
average the seeds back per flavor. The seed varies only pretraining (the finetune seed is
held fixed), so the spread it reveals is foundation-initialization variance. See
[[Finetune Protocols]].

## Screening scale

The 250K corpus is the screening set, not the final training scale. It is large enough to rank
the flavors cheaply and decide which descriptor targets earn further cost. Any flavor that beats
the baseline here is the candidate for a full-scale pretrain on the 1M corpus (the scale
[[Stock CheMeleon]] uses) to produce the release foundation, holding this same regime fixed so
the 1M foundation is the screened experiment at scale rather than a new one.

## Why it matters

A fit-to-purpose comparison only means something if the foundations differ in exactly one
way. Any drift in corpus, featurizer, aggregation, or regime would confound the report card.

## Prescaling recipe vs. corpus size (open question)

The [[Prescaling Ablation]] triage that picked `chemeleon_baseline` ran on the full corpus,
not this regime's 250K corpus, because the training-collapse regime fix moved the triage to
the full corpus to eliminate corpus size as a confound at the same time. A follow-up rerun of
the full triage on the 250K corpus found a different winning recipe per protocol (frozen
winner `plus_drop_low_var`, not `chemeleon_baseline`, which drops to 5th of 7), so the recipe
this regime bakes in was picked on a corpus larger than the one it is actually applied to.
This does not block the flavor sweep (frozen is still the sweep's protocol, and
`chemeleon_baseline` is what `split.py` already does with no code change needed either way),
but it is a known gap between the triage's basis and this regime's actual corpus; revisit if
the flavor sweep's report card looks off relative to expectations.

## Related

- Center of the flavor graph; every flavor branches off this node.
- Compared against the published [[Stock CheMeleon]] (different corpus and regime).
