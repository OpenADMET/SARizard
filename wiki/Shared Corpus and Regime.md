---
tags: [method, status/blue]
---
# Shared Corpus and Regime

> **Summary:** The one thing held constant across every flavor. A fixed 944,296-molecule corpus
> from the original CheMeleon PubChem set, a single pretraining seed, and one pretraining regime
> (capped epochs, shared LR schedule, mean aggregation, default graph featurizer). The only
> intended difference between flavors is the target block (and MSE vs BCE for binary targets),
> so the [[Report Card]] columns are comparable.

## What is fixed

- Corpus: `corpus/corpus_full.parquet`, 944,296 molecules from the Zenodo PubChem set. This
  superseded the 250K screening subset (`corpus/corpus_250k.parquet`) when the whole sweep moved
  to full scale; the 250K flavor-sweep artifacts are archived at `archive/flavor_sweep_250k/`.
  The one exception is [[surrogate_adme]], whose target is only defined on its Novartis
  molecules, so it pretrains on its own corpus and is a different-corpus reference arm rather
  than a comparable column.
- Featurizer: chemprop DEFAULT (`MultiHotAtomFeaturizer.v2` + `MultiHotBondFeaturizer`), so
  `d_v`/`d_e` match openadmet's finetuning featurizer.
- Aggregation: mean (openadmet rebuilds foundations with mean aggregation).
- Regime: epoch cap, patience, LR schedule, and model width all in `sarizard/pretraining/config.py`.
- Storage chunking is fixed across flavors, so the train/val split selects the same molecules
  for every flavor and the pretraining batch size is identical.
- Descriptor prescaling: one recipe (`chemeleon_baseline`), chosen once by the
  [[Prescaling Ablation]] triage and applied identically to every continuous flavor. The
  corpus-size caveat this entry used to carry is resolved: the triage ran on the full corpus and
  the sweep now runs on the same one. A different caveat replaces it, that the recipe was picked
  on single-seed data the 5-seed redo did not re-test; see [[Prescaling Ablation]].
- Target dropout: the masked-pretext keep fraction is fixed across flavors, with one hard
  invariant. Any target block at or under 30 dims (`DROPOUT_OVERRIDE_MAX_DIM`: [[jazzy]] at 6,
  [[surrogate_adme]] at 25) pretrains at `dropout_fraction=0.0`, because a fixed keep fraction
  starves supervision at narrow widths. `train.py` resolves this from the split before reading
  the `--dropout-fraction` flag and rejects a nonzero override for such a flavor. Settled, not
  an open ablation.

## Seeds

One pretraining seed per flavor (42), five finetune seeds off it. The regime pins the
pretraining seed so each flavor is exactly one foundation, and replication happens on the
finetune side instead: seeds 1-5 finetune off that fixed foundation, so a report-card error bar
is **finetune** variance, not foundation-initialization variance. See [[Finetune Protocols]].

The machinery can pretrain a flavor at several seeds (`FLAVOR_SEEDS`, tagged
`<flavor>__s<seed>`), which would measure initialization noise instead, but the sweep does not
use it that way: pretraining 15 flavors at several seeds each costs far more than finetuning
does, and the finetune spread is the noise floor a per-flavor difference has to clear in
practice.

## Scale

The corpus is the release training scale, not a screen. The study originally planned to rank
flavors cheaply on 250K and scale only the winners to 1M; that two-stage plan was folded into
the sweep itself, which runs directly at full scale, so every flavor's foundation is already the
release-scale artifact. The remaining headroom is the difference between this corpus (944,296)
and the full 1M CheMeleon PubChem set, which no result so far argues is worth spending.

## Why it matters

A fit-to-purpose comparison only means something if the foundations differ in exactly one
way. Any drift in corpus, featurizer, aggregation, or regime would confound the report card.

## Prescaling recipe vs. corpus size (resolved)

This was an open gap while the triage ran on the full corpus and the sweep ran on 250K: a
follow-up rerun of the triage at 250K found a different winning recipe per protocol
(`plus_drop_low_var` frozen, with `chemeleon_baseline` dropping to 5th of 7), so the recipe
baked in had been picked on a corpus larger than the one it was applied to. Moving the sweep to
the full corpus closed it; triage basis and sweep corpus now match.

The finding that produced the gap still stands and is worth keeping: **the prescaling ranking is
corpus-size sensitive**, not merely noisier at smaller scale. Do not assume the recipe ranking
transfers to a different corpus size, including a future move to the full 1M set.

## Related

- Center of the flavor graph; every flavor branches off this node.
- Compared against the published [[Stock CheMeleon]] (different corpus and regime).
