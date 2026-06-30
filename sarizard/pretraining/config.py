# Shared pretraining regime. Held fixed across every flavor so the report-card columns
# are apples-to-apples; the only intended per-flavor difference is the target block (and
# MSE vs BCE for binary targets). Tune the regime here, not per flavor.

# for data preparation
WINSORIZATION_FACTOR = 6

# Target caching. Storage chunk rows are fixed across every flavor so the chunk-based
# train/val split (split.py) holds out the same molecules for each flavor, and the
# pretraining batch size (CORPUS_CHUNK_ROWS * CHUNKS_PER_BATCH) is identical across
# flavors; only the target block differs. This trades the upstream "~1 MB per chunk"
# convention (which made byte size uniform but molecule count per batch vary with
# target width) for a uniform molecule batch. Compute block rows is the larger,
# decoupled granularity at which target calculators run, for throughput.
CORPUS_CHUNK_ROWS = 128
COMPUTE_BLOCK_ROWS = 2_048

# 3D conformer generation for the 3D flavors (usrcat, whim, e3fp), recorded for
# reproducibility. Conformer targets are only approximately reproducible across runs.
CONFORMER_SEED = 42
CONFORMER_NUM = 1
CONFORMER_FORCE_FIELD = "MMFF94"

# for training
DROPOUT_FRACTION = 0.30
EPOCHS = 100
PATIENCE = 5
INITIAL_LEARNING_RATE = 0.0001
MAXIMUM_LEARNING_RATE = 0.001
FINAL_LEARNING_RATE = 0.0001
WARMUP_EPOCHS = 5
CHUNKS_PER_BATCH = 2

# model hyperparameters
FNN_HIDDEN_SIZE = 2_048
FNN_HIDDEN_LAYERS = 1
FNN_ACTIVATION = "LEAKYRELU"  # one of: RELU, LEAKYRELU, PRELU, TANH, ELU
MP_HIDDEN_SIZE = 2_048
MP_DEPTH = 6
MP_ACTIVATION = "LEAKYRELU"  # one of: RELU, LEAKYRELU, PRELU, TANH, ELU
# Compatibility invariant: openadmet-models finetunes with its ChemPropFeaturizer
# (chemprop DEFAULT: MultiHotAtomFeaturizer.v2 + MultiHotBondFeaturizer). The message
# passing input dims are baked into the exported checkpoint, so pretraining must use the
# same featurizer. RIGR would change d_v/d_e and break loading via from_foundation.
FEATURIZER = "DEFAULT"  # must stay DEFAULT for openadmet compatibility (see AGENTS.md)
