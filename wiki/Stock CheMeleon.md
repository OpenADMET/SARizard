---
tags: [backbone, status/baseline]
---
# Stock CheMeleon

> **Summary:** The published CheMeleon foundation: a D-MPNN pretrained to regress 1613 Mordred
> descriptors over 1M PubChem molecules. It is an external reference column, not one of our
> flavors, because it uses a different corpus and a fuller regime than [[Shared Corpus and Regime]].

## What it is

The off-the-shelf foundation openadmet loads with `from_foundation: chemeleon`. The committed
`configs/_baseline/` recipes use it unchanged, so finetuning them gives a stock-CheMeleon
column for the [[Report Card]] at no extra pretraining cost.

## Why it is a reference, not an arm

Our flavors are pretrained on a fixed 250K subset under a reduced shared regime to isolate the
effect of the target block. Stock CheMeleon was trained on 1M molecules with a different
target and schedule, so it is not apples-to-apples; read it as a strong external baseline.

## Related

- Contrast with [[Shared Corpus and Regime]] (the controlled arms).
- The closest in-study flavor by target is [[osmordred]] (an osmordred reimplementation of the
  Mordred block stock CheMeleon regresses).
