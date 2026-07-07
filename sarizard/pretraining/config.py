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
# DROPOUT_FRACTION, PATIENCE, WARMUP_EPOCHS, and FNN_HIDDEN_SIZE below were reconciled against
# ../foundation-models/pretraining/run_pretraining.py after every prescaling-ablation run
# diverged mid-pretraining (loss/R2 blowing up 3-10 epochs in). That sibling implementation
# trains the same MPNN/descriptor-regression task without this instability, so its regime is
# adopted as canonical here; GRADIENT_CLIP_VAL is added on top since neither implementation
# had it.
DROPOUT_FRACTION = 0.85  # keeps 15% of targets/step (matches sibling's MASKING_RATIO=0.15);
# was 0.30 (keeps 70%), a much denser per-step supervision load on a 3585-dim target block

# a target block at or below this width keeps under ~5 elements/step at DROPOUT_FRACTION
# (ml_qm 24, surrogate_adme 25, jazzy 6 all qualify); train.py drops dropout to 0.0 for these
# automatically unless --dropout-fraction is passed explicitly, rather than pretraining
# against near-zero supervision density. Not the ablation once planned to pick this value
# (TODO.md Future experiments): an explicit override decision instead, since a 24-25 dim
# target is unlikely to need masking's co-adaptation guard the way osmordred's 3585 does.
DROPOUT_OVERRIDE_MAX_DIM = 30
EPOCHS = 100
PATIENCE = 50  # was 5; the sibling's patience is 10x more tolerant of a transient bad epoch
INITIAL_LEARNING_RATE = 0.0001
MAXIMUM_LEARNING_RATE = 0.001
FINAL_LEARNING_RATE = 0.0001
WARMUP_EPOCHS = 2  # was 5; chemprop's own MPNN default, and what the sibling implicitly uses
GRADIENT_CLIP_VAL = 0.5  # new: neither this repo nor the sibling clipped gradients before
CHUNKS_PER_BATCH = 2

# model hyperparameters
FNN_HIDDEN_SIZE = 1_024  # was 2_048; matches the sibling's PREDICTOR_HIDDEN_DIM
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
