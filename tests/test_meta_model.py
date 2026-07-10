"""Tests for the meta-model's prediction collection and per-seed scoring."""

import json

import numpy as np
import pandas as pd

from sarizard.analysis.meta_model import collect_predictions, run


def _write_result(root, variant_label, endpoint, y_true, preds, cols):
    """Write a minimal anvil-style result dir with cached predictions and the target sidecar."""
    data_dir = root / variant_label / endpoint / "data"
    data_dir.mkdir(parents=True)
    np.save(data_dir / "y_pred.npy", preds)
    pd.DataFrame({cols[0]: y_true}).to_csv(data_dir / "y_test.csv", index=False)
    (data_dir / "target_cols.json").write_text(json.dumps(cols))


def test_collect_predictions_keeps_seeds_in_separate_buckets(tmp_path):
    cols = ["logd"]
    y_true = [1.0, 2.0, 3.0]
    # two seeds of one flavor, predicting all-0 and all-2 respectively
    _write_result(tmp_path, "ecfp__s1", "cyp_mt", y_true, np.zeros((3, 1)), cols)
    _write_result(tmp_path, "ecfp__s2", "cyp_mt", y_true, np.full((3, 1), 2.0), cols)

    store = collect_predictions(tmp_path, ["ecfp__s1", "ecfp__s2"])

    # the two seeds stay in separate buckets rather than being averaged into one vector, so
    # each seed's predictions survive intact for its own scoring pass
    (by_seed,) = store.values()
    assert set(by_seed) == {1, 2}
    assert np.allclose(by_seed[1]["preds"]["ecfp"], [0.0, 0.0, 0.0])
    assert np.allclose(by_seed[2]["preds"]["ecfp"], [2.0, 2.0, 2.0])


def test_collect_predictions_strips_lr_mode_prefix_to_bare_flavor(tmp_path):
    cols = ["logd"]
    y_true = [1.0, 2.0, 3.0]
    # an LR-protocol result dir is namespaced lr_<mode>__<flavor>__s<seed>
    _write_result(tmp_path, "lr_reduced__ecfp__s1", "cyp_mt", y_true, np.zeros((3, 1)), cols)

    store = collect_predictions(tmp_path, ["lr_reduced__ecfp__s1"], strip_prefix="lr_reduced__")

    # the prefix is stripped so the feature key is the bare flavor the registry filter expects,
    # not lr_reduced__ecfp which would never match flavor_names() in _evaluate_endpoint
    (by_seed,) = store.values()
    assert set(by_seed[1]["preds"]) == {"ecfp"}


def test_run_handles_seeds_with_different_test_lengths(tmp_path):
    # regression: the multi-task endpoints resample their test set per finetune seed, so two
    # seeds have different numbers of test rows. Averaging raw predictions across seeds used to
    # np.stack these and raise "all input arrays must have the same shape"; scoring per seed and
    # averaging metrics must instead produce a row without raising.
    cols = ["logd"]
    rng = np.random.default_rng(0)
    for seed, n in ((1, 40), (2, 55)):
        y = rng.normal(size=n)
        # two flavors sharing this seed's split: one tracks y, one is noise, so the stacker has
        # a real signal to combine and _evaluate_endpoint returns a score
        for flavor, preds in (
            ("ecfp", (y + rng.normal(scale=0.1, size=n)).reshape(n, 1)),
            ("rdkit2d", rng.normal(size=n).reshape(n, 1)),
        ):
            _write_result(tmp_path, f"{flavor}__s{seed}", "cyp_mt", y.tolist(), preds, cols)

    frame = run(tmp_path, ["ecfp__s1", "ecfp__s2", "rdkit2d__s1", "rdkit2d__s2"], "lgbm", 5, 42)

    assert len(frame) == 1
    (row,) = frame.to_dict("records")
    assert row["n_seeds"] == 2
    assert row["meta_r2_std"] >= 0.0
