---
tags: [method, status/blue]
---
# Report Card

> **Summary:** The north-star artifact. Endpoints as rows, foundation flavors as columns,
> rendered as a fixed pair of cards per finetune protocol: an R² card and an MAE %-change card
> against [[Stock CheMeleon]]. It answers: which pretraining target serves which endpoint and
> endpoint family best.

## How it is built

- `sarizard/analysis/evaluate.py` reloads each finetuned model, predicts on the held-out test split,
  and writes per-(flavor, endpoint) metrics to `results/metrics.csv` for frozen and
  `results/lr_metrics.csv` for the other two protocols, caching predictions for the
  [[Meta-Model]].
- `sarizard/analysis/report_card.py` renders **both** cards per call into `plots/`; there is no
  `--metric` or `--color-mode` flag. Pass `--lr-mode {reduced,unlocked}` to read the LR CSV and
  strip the `lr_<mode>__` prefix back to bare flavor names, and `--baseline-flavor
  chemeleon_stock_<mode>` so each card diverges around its own protocol's baseline.
- Each flavor's `<flavor>__s<seed>` variants collapse into a single averaged column, so a flavor
  is one column regardless of seed count (see [[Finetune Protocols]]).

## The two cards

- **R² card**: fixed red-to-green scale (red 0, green 1), the stock baseline as the first column
  behind a spacer, an AVERAGE row meaning each column across endpoints, and every cell annotated
  with its ± seed standard deviation.
- **MAE %-change card**: `100·(mae_flavor − mae_baseline)/mae_baseline`, green where the flavor's
  MAE is lower. A cell is **colored only where the difference is significant** (p ≤ 0.05);
  non-significant cells are white and annotated with their p-value, so the card shows only what
  the seed spread supports.

Significance uses **Dunnett's test, with one endpoint row as one comparison family**. The row's
flavors are all measured against the same stock baseline, so they are corrected together and each
p-value is already family-wise. This replaced an uncorrected per-cell Welch t-test, which ran 480
independent tests per card and let false positives scale with the number of flavors shown; the
change costs the frozen card 143 colored cells down to 95. Correcting per row and not across the
whole card is deliberate, so error across the 32 rows is not controlled: scanning the entire card
for the single best cell is still an uncontrolled search.

The **AVERAGE row runs a second, separate Dunnett family**, since summarizing a flavor across all
32 endpoints is a different question from any single cell. Each group is one value per finetune
seed: that seed's mean MAE %-change across the card's endpoints, with the baseline put through
the same aggregation as the control. The row was previously colored by its mean change no matter
what; it is now whitened when the seed spread does not separate it from the baseline, which
leaves 7 of 15 flavors colored under frozen, 7 under reduced, and 4 under unlocked. See
`FINDINGS.md` for the caveats,
notably that the pooled-variance assumption cannot be checked at five seeds and that no
correction touches the single-pretraining-seed limitation.

## Reading it

- Rows group by source dataset with a bold separator and label. Where one (dataset, endpoint)
  pair comes from more than one recipe, the recipe is appended to the row label rather than the
  two being averaged together.
- Binary fingerprint flavors are a leaky, weak pretext; a poor column for them is a result.
- A single strong cell is weak evidence on its own. The [[PXR External Test]] found that the
  card's cleanest specialization signal, [[rdkit2d]] on PXR, did not reproduce on a fixed
  external hold-out, because that endpoint's internal split moves with the finetune seed.

## Result

All 15 flavors, 5 seeds, all three protocols. Frozen leaders: [[surrogate_adme]] 0.370,
[[minimol]] 0.343, [[rdkit2d]] 0.323, against a [[Stock CheMeleon]] baseline of 0.294. Four
flavors clear stock significantly under frozen and eight under reduced, but under unlocked the
stock baseline rises to 0.337 and only [[surrogate_adme]] still clears it, with eleven flavors
significantly below. Reduced is the protocol where pretraining pays; unlocked overwrites it.

The family read is more sobering than the fit-to-purpose premise expected: the pooled flavor mean
beats stock on only three of eight families, and those three (clearance, hERG, CYP inhibition)
are the hardest, where everything sits near the noise floor. Descriptor pretraining buys a
per-endpoint best-of, not a family-level specialization. Full tables in `FINDINGS.md`.

## Related

- Built from results of every flavor finetuned over [[Clearance]], [[Permeability]],
  [[Solubility]], [[Lipophilicity]], [[Potency]], [[CYP Inhibition]], and [[hERG]].
- Paired artifact: [[Meta-Model]].
- Studies that sit beside it rather than on it: [[osmordred_surrogate]],
  [[External Foundations]], [[PXR External Test]].
