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

## Provenance and discipline

These tables are already standardized (canonical SMILES, log-scale potency targets).
Potency endpoints are in log space; do not average or model raw concentrations. Splits
are structure-aware (cluster or predefined) to respect applicability-domain evaluation.
Regenerate by copying from the source project or its upstream data sources.
