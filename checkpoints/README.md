# SARizard pretrained foundations

Twenty-seven CheMeleon-format message-passing checkpoints: the twenty-three pretrained in this
repository, plus four external reference checkpoints that were not (see below). Each file is the
converted foundation export, not a Lightning training checkpoint, so it carries model weights
only and cannot be used to resume pretraining.

Every file is seed 42. `MANIFEST.csv` lists the target block, target width, loss, target-dropout
fraction, pretraining corpus, source, size, and sha256 of each one.

## Getting the weights

The checkpoints are **release assets, not files in this repository**: 898 MB of binaries would
sit in the git history of every clone forever. A clone gives you this README, `MANIFEST.csv`, and
`SHA256SUMS`; the weights come from the repository's GitHub releases page.

Download them into this directory, preserving the category subdirectories below, and the paths in
`MANIFEST.csv` and `SHA256SUMS` will resolve as written.

## What these are

SARizard pretrains "flavors" of the CheMeleon molecular foundation model that differ only in the
descriptor block each is regressed against, then finetunes and evaluates each on ADMET benchmark
endpoints. The backbone, corpus, and pretraining regime are held fixed across flavors, so the
only intended difference between two `flavors/` checkpoints is the target block. Results and the
read on each flavor are in `FINDINGS.md` at the repository root.

## Layout

Category directories, as the release assets are packaged:

| directory | count | what |
|---|---|---|
| `flavors/` | 15 | the report-card foundations, one per descriptor target |
| `prescaling_ablations/` | 7 | one osmordred target, seven descriptor-preprocessing recipes |
| `controls/` | 1 | `osmordred_surrogate`, the chemical-space control |
| `external_reference/` | 4 | **not pretrained here**, included for comparison only |

## Loading

The format is what `openadmet-models` expects from `torch.load(path, weights_only=True)`: a dict
of `{"hyper_parameters": <BondMessagePassing kwargs>, "state_dict": <message-passing state>}`.

```python
import torch

ckpt = torch.load("flavors/rdkit2d__s42.pt", weights_only=True)
assert set(ckpt) == {"hyper_parameters", "state_dict"}
```

In `openadmet-models`, load one through `ChemPropModel.from_foundation`. Two properties are baked
into every checkpoint and must match at load time: aggregation is `MeanAggregation`, and the
graph featurizer is the openadmet `DEFAULT` (`MultiHotAtomFeaturizer.v2` +
`MultiHotBondFeaturizer`), not `RIGR`. A mismatch degrades the foundation silently rather than
raising.

## Read this before comparing checkpoints

**Two flavors are not on the shared corpus.** Every `flavors/` checkpoint pretrains on the same
944,296-molecule corpus except `surrogate_adme`, whose target (25 Novartis ADME predictions) is
only defined on the Novartis molecules, so it pretrains on those. `controls/osmordred_surrogate`
is on that same Novartis corpus by design: it holds the target identical to `flavors/osmordred`
and changes only the corpus, to separate chemical space from target. Neither is an
apples-to-apples comparison against the other thirteen. The `corpus` column in `MANIFEST.csv`
says which corpus each file used.

**The external reference checkpoints were not pretrained in this repository.** `molpile_1M`,
`molpile_5M`, `molpile_10M`, and `expansion_gen` were produced elsewhere and copied in to compare
pretraining corpus and size against the flavors. They are included here because they were part of
the comparison, not as SARizard outputs. Their upstream authorship, training data, and licensing
were not established by this project, and the `source` column in `MANIFEST.csv` records only the
local path each was copied from. Establish those terms before redistributing them further.

**Target dropout is zero below thirty dimensions.** The masked-pretext objective keeps a fixed
fraction of target dimensions, which starves supervision at narrow target widths, so any target
block at or under 30 dimensions pretrains at `dropout_fraction=0.0`. That is why `jazzy` (6) and
`surrogate_adme` (25) differ from the 0.85 the others use. This is a deliberate invariant, not an
inconsistency between runs.

**The prescaling ablations are not alternative flavors.** All seven share one osmordred target
and one backbone and differ only in descriptor preprocessing, so they are useful for studying
preprocessing and misleading if read as seven different foundations.
`ablation_chemeleon_baseline` was later dropped from the ablation registry and has no five-seed
results, though the checkpoint itself is sound.

## Not included

`ml_qm` is omitted. Its qmdesc target legitimately carries about 1.4% NaN, and the resulting
foundation came back all-NaN in every tensor; it loads without error and produces garbage. The
flavor was dropped from the study.

Checkpoints from superseded runs are also omitted: the 250K-molecule screening sweep, the
full-corpus ablation rerun, and the pre-gradient-clip ablations whose pretraining diverged. They
remain under `archive/` in the repository.

## Verifying a download

```bash
sha256sum -c SHA256SUMS
```

`SHA256SUMS` carries the same digests as `MANIFEST.csv` in the standard format; the CSV is for
reading the metadata, the sums file for checking the bytes.
