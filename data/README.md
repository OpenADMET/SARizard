# Benchmark data

The benchmark datasets and their splits are copied from the sibling project
`../information-gain-metric/data/`. They are gitignored here (not redistributed); this
file records what they are and where they came from so they can be regenerated or obtained.

## Datasets

| File | Source | Endpoints |
|---|---|---|
| `asap.csv` | ASAP | CLint HLM/MLM, MDR1, LogD, KSOL |
| `asap_potency.csv` | ASAP | MERS pIC50, SARS2 pIC50 |
| `biogen.csv` | Biogen | CLint HLM/RLM/MLM, MDR1, SOL |
| `chembl.csv` | ChEMBL | CLint HLM/RLM/MLM |
| `expansion_data_train.csv`, `expansion_data_test.csv` | ExpansionRx | CLint HLM/MLM, Caco2 Papp/Efflux, LogD, KSOL, MPPB, MBPB |
| `ChEMBL_IC50_CYP_multitask_ChEMBL37.parquet` | ChEMBL 37 | CYP3A4/2C9/2D6/1A2 IC50 |
| `ChEMBL_IC50_HERG_CHEMBL240_aggregated.parquet` | ChEMBL 240 | hERG pIC50 |
| `pxr_pec50.parquet` | Octant | PXR pEC50 |

`splits/` holds the predefined train/val/test split files (both cluster and random
validation strategies where applicable), matching the split discipline used in the
baseline recipes.

### PXR external-test split (reduced-protocol rerun)

The standard PXR endpoint splits `pxr_pec50.parquet` with an inline Butina `ClusterSplitter`, so
the split moves with the finetune seed. The `pxr_ext` rerun instead evaluates on two external,
fixed held-out test sets from the OpenADMET PXR challenge:

| File | Source | Rows | Notes |
|---|---|---|---|
| `splits/pxr_ext_train.csv`, `splits/pxr_ext_val.csv` | derived from `pxr_pec50.parquet` | 1950 / 217 | one fixed 90/10 split (`np.random.default_rng(42)`), shared across every flavor and seed |
| `splits/pxr_test_phase1.csv` | HF `openadmet/pxr-challenge-train-test` (`pxr-challenge_TEST_PHASE_1_UNBLINDED.csv`) | 253 | external held-out test |
| `splits/pxr_test_phase2.csv` | HF `openadmet/pxr-challenge-train-test` (`pxr-challenge_TEST_PHASE_2_UNBLINDED.csv`) | 260 | external held-out test |

Regenerate with `python -m sarizard.analysis.build_pxr_ext_splits` (needs a Hugging Face token for
the gated dataset). The challenge files already carry the target as a `pEC50` column in the same
`-log10(molarity)` convention as the training `PXR_pEC50`, so it is renamed and used as-is with no
log transform; test SMILES are RDKit-canonicalized into `OPENADMET_CANONICAL_SMILES`. No test
molecule overlaps the training set by InChIKey, and the two phases are disjoint.

## Provenance and discipline

These tables are already standardized (canonical SMILES, log-scale potency targets).
Potency endpoints are in log space; do not average or model raw concentrations. Splits
are structure-aware (cluster or predefined) to respect applicability-domain evaluation.
Regenerate by copying from the source project or its upstream data sources.
