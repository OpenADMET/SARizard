---
tags: [control, status/complete]
---
# External Foundations

> **Summary:** A comparison of pretraining datasets rather than descriptor blocks. Four
> externally pretrained CheMeleon-format foundations, carrying no target or pretraining in this
> repo, are finetuned on the same 24 endpoints under the same protocols as the flavor sweep and
> measured against [[Stock CheMeleon]]. It asks whether a bigger or different pretraining corpus
> beats the published backbone, where the flavor sweep asks whether a different target does.

- Foundations: `molpile_1M`, `molpile_5M`, `molpile_10M` (from
  `/home/westd1/myscratch/foundation-models/datafiles/foundation_models/`) and `expansion_gen`
  (from `/home/westd1/myscratch/202606_generative_foundation_models/expansion_gen/`)
- 1440 finetunes: 4 foundations × 24 endpoints × 5 seeds × 3 protocols
- Standalone by construction: dedicated `results/external_metrics.csv` and its own cards in
  `plots/external_foundations/`, never the shared `results/metrics.csv` or the [[Report Card]]

## How it runs

The flavor sweep's finetune-only path pointed at foreign checkpoints. Each is copied to
`foundations/<name>__s42_mp.pt` (a copy, not a symlink, so recipe generation can resolve it
relative to the repo root), then passed through the same gate the repo's own foundations meet:
the openadmet `{hyper_parameters, state_dict}` format, and message-passing dims (`d_v`, `d_e`)
matching an existing repo foundation. See [[Shared Corpus and Regime]] for why those dims are a
hard invariant. The driver is `slurm/run_external_foundations.sh`; the existing 5-seed stock
baseline is reused as the reference column rather than re-finetuned.

## Result

Mean R² across the 32 endpoint-columns, per seed then averaged over seeds. The frozen stock
column is 6 seeds here (the legacy 42 plus 1-5) against the [[Report Card]]'s 5, which accounts
for a 0.001 difference between the two:

| foundation | frozen | reduced | unlocked |
|---|---|---|---|
| **[[Stock CheMeleon]]** | **0.295 ± 0.009** | **0.316 ± 0.014** | **0.337 ± 0.008** |
| molpile_5M | 0.255 ± 0.009 | 0.264 ± 0.015 | 0.242 ± 0.025 |
| molpile_10M | 0.220 ± 0.013 | 0.250 ± 0.009 | 0.213 ± 0.007 |
| molpile_1M | 0.217 ± 0.015 | 0.241 ± 0.022 | 0.292 ± 0.019 |
| expansion_gen | 0.186 ± 0.013 | 0.250 ± 0.016 | 0.231 ± 0.015 |

**Every external foundation loses to stock under every protocol, and all twelve deficits are
significant** (Welch against the same-protocol stock seeds, worst p 0.003). The largest
deficits are `expansion_gen` frozen (−0.109) and `molpile_10M` unlocked (−0.124).

**Size does not buy accuracy monotonically.** `molpile_5M` leads `1M` and `10M` under frozen and
reduced, but under unlocked the order inverts to `1M` first and `10M` last. The spread between
the three sizes (0.04-0.08) is the same scale as the deficit to stock, so a 10x corpus increase
is not visible as a consistent gain.

## How much this can carry

Less than the flavor sweep. The four checkpoints differ in corpus, size, and pretraining recipe
at once, so a per-foundation delta cannot be attributed to corpus size alone; this is not the
controlled one-variable design the flavor sweep holds itself to. What it does support is a
negative claim worth having: stock CheMeleon is a strong baseline that more pretraining
molecules do not beat, which sits alongside the sweep's finding that beating it takes a
well-chosen target.

## Related

- Reference column: [[Stock CheMeleon]] · Protocols: [[Finetune Protocols]]
- The comparison it complements: [[Report Card]] (targets held variable, backbone fixed)
- Other standalone studies: [[osmordred_surrogate]], [[PXR External Test]]
