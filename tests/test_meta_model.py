"""Tests for the meta-model's prediction collection, focused on seed averaging."""

import json

import numpy as np
import pandas as pd

from sarizard.analysis.meta_model import collect_predictions


def _write_result(root, variant_label, endpoint, y_true, preds, cols):
    """Write a minimal anvil-style result dir with cached predictions and the target sidecar."""
    data_dir = root / variant_label / endpoint / "data"
    data_dir.mkdir(parents=True)
    np.save(data_dir / "y_pred.npy", preds)
    pd.DataFrame({cols[0]: y_true}).to_csv(data_dir / "y_test.csv", index=False)
    (data_dir / "target_cols.json").write_text(json.dumps(cols))


def test_collect_predictions_averages_seed_replicates(tmp_path):
    cols = ["logd"]
    y_true = [1.0, 2.0, 3.0]
    # two seeds of one flavor, predicting all-0 and all-2 respectively; mean is all-1
    _write_result(tmp_path, "ecfp__s1", "cyp_mt", y_true, np.zeros((3, 1)), cols)
    _write_result(tmp_path, "ecfp__s2", "cyp_mt", y_true, np.full((3, 1), 2.0), cols)

    store = collect_predictions(tmp_path, ["ecfp__s1", "ecfp__s2"])

    # the two seed variants collapse to a single base-flavor feature, molecule-wise averaged
    (entry,) = store.values()
    assert set(entry["preds"]) == {"ecfp"}
    assert np.allclose(entry["preds"]["ecfp"], [1.0, 1.0, 1.0])
