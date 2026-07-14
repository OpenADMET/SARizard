"""Tests for the toggleable descriptor prescaling pipeline.

Each test builds a small synthetic target store with the pathologies the pipeline is meant
to handle (NaNs, infs, a heavy outlier, a constant column, a correlated pair) and checks one
behavior. The store uses the production chunk size so the train/val chunk split is realistic.
"""

import numpy as np
import pytest
import zarr

from sarizard.pretraining.prescaling import PrescalingConfig, get_ablation, run_prescaling
from sarizard.pretraining.splitting import train_val_chunk_indices

CHUNK_ROWS = 128
N_CHUNKS = 40
N_ROWS = CHUNK_ROWS * N_CHUNKS
N_COLS = 8

# column roles in the synthetic store, asserted against by index
COL_CORR_SOURCE = 0
COL_CORR_COPY = 1  # ~ perfectly correlated with col 0
COL_CONSTANT = 2  # zero variance
COL_SOME_NAN = 3
COL_SOME_INF = 4
COL_OUTLIER = 5  # one extreme value
COL_PLAIN = 6
COL_ALL_NAN = 7  # no finite values at all

ROW_FAILED = 200  # an all-NaN row standing in for a molecule the calculator failed on


def _build_store(path) -> zarr.Array:
    """Write the synthetic target store and return the opened array."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N_ROWS, N_COLS)).astype("float32")
    x[:, COL_CORR_COPY] = x[:, COL_CORR_SOURCE] + rng.normal(scale=1e-3, size=N_ROWS)
    x[:, COL_CONSTANT] = 3.14
    x[:50, COL_SOME_NAN] = np.nan
    x[60:65, COL_SOME_INF] = np.inf
    x[70:75, COL_SOME_INF] = -np.inf
    x[:, COL_OUTLIER] = rng.normal(size=N_ROWS) * 100.0
    x[0, COL_OUTLIER] = 1e6
    x[:, COL_ALL_NAN] = np.nan
    x[ROW_FAILED, :] = np.nan

    store = zarr.create_array(
        store=str(path),
        shape=(N_ROWS, N_COLS),
        chunks=(CHUNK_ROWS, N_COLS),
        dtype="float32",
        compressors=None,
        fill_value=np.nan,
    )
    store[:] = x
    return store


@pytest.fixture
def target_store(tmp_path):
    """A synthetic raw target.zarr exercising every pipeline pathology."""
    path = tmp_path / "target.zarr"
    _build_store(path)
    return path


@pytest.mark.parametrize(
    "ablation",
    ["minimal", "order_fix", "plus_drop_corr",
     "plus_drop_low_var", "plus_yeo_johnson", "full"],
)
def test_nonfinite_inputs_are_not_imputed(target_store, tmp_path, ablation: str):
    out = tmp_path / f"{ablation}.zarr"
    raw = zarr.open_array(str(target_store), mode="r")[:]

    summary = run_prescaling(target_store, out, get_ablation(ablation), force=True)

    # the output cell is non-finite exactly where the corresponding kept-column input cell
    # was non-finite: finite inputs scale to finite outputs, and non-finite inputs re-scatter
    # back to NaN rather than being imputed (the masked loss skips them per element)
    result = zarr.open_array(str(out), mode="r")[:]
    dropped = (
        set(summary["dropped_invalid"])
        | set(summary["dropped_corr"])
        | set(summary["dropped_low_var"])
    )
    kept = [col for col in range(N_COLS) if col not in dropped]
    assert np.array_equal(~np.isfinite(raw[:, kept]), ~np.isfinite(result))


@pytest.mark.parametrize(
    "ablation",
    ["minimal", "order_fix", "plus_drop_corr",
     "plus_drop_low_var", "plus_yeo_johnson", "full"],
)
def test_failed_molecule_row_stays_nan(target_store, tmp_path, ablation: str):
    out = tmp_path / f"{ablation}.zarr"

    run_prescaling(target_store, out, get_ablation(ablation), force=True)

    # a molecule whose target row is all NaN must remain all NaN so the masked pretraining
    # loss skips it rather than training on re-scattered zeros (the B1 re-scatter regression)
    result = zarr.open_array(str(out), mode="r")[:]
    assert np.isnan(result[ROW_FAILED]).all()


def test_all_nan_column_is_always_dropped(target_store, tmp_path):
    summary = run_prescaling(
        target_store, tmp_path / "out.zarr", get_ablation("minimal"), force=True
    )

    assert COL_ALL_NAN in summary["dropped_invalid"]


def test_correlated_column_dropped_only_under_drop_corr(target_store, tmp_path):
    without = run_prescaling(
        target_store, tmp_path / "a.zarr", get_ablation("order_fix"), force=True
    )
    with_corr = run_prescaling(
        target_store, tmp_path / "b.zarr", get_ablation("plus_drop_corr"), force=True
    )

    assert without["dropped_corr"] == []
    assert COL_CORR_COPY in with_corr["dropped_corr"]


def test_constant_column_dropped_only_under_low_var(target_store, tmp_path):
    without = run_prescaling(
        target_store, tmp_path / "a.zarr", get_ablation("order_fix"), force=True
    )
    with_lv = run_prescaling(
        target_store, tmp_path / "b.zarr", get_ablation("plus_drop_low_var"), force=True
    )

    assert without["dropped_low_var"] == []
    assert COL_CONSTANT in with_lv["dropped_low_var"]


def test_full_drops_more_columns_than_order_fix(target_store, tmp_path):
    order_fix = run_prescaling(
        target_store, tmp_path / "a.zarr", get_ablation("order_fix"), force=True
    )
    full = run_prescaling(
        target_store, tmp_path / "b.zarr", get_ablation("full"), force=True
    )

    assert full["n_cols_out"] < order_fix["n_cols_out"]


def test_winsorization_bounds_the_outlier(target_store, tmp_path):
    # the legacy production recipe: std-based winsorize and z-score both fit on the raw target
    std_baseline = PrescalingConfig(
        name="std_baseline",
        do_winsorize=True,
        winsorize_method="std",
        winsorize_std_factor=6.0,
        zscore_fit="raw",
    )
    run_prescaling(target_store, tmp_path / "min.zarr", get_ablation("minimal"), force=True)
    run_prescaling(target_store, tmp_path / "base.zarr", std_baseline, force=True)
    run_prescaling(target_store, tmp_path / "of.zarr", get_ablation("order_fix"), force=True)

    # nan-aware: the failed-molecule row is NaN in every output store
    min_max = np.nanmax(np.abs(zarr.open_array(str(tmp_path / "min.zarr"), mode="r")[:]))
    base_max = np.nanmax(np.abs(zarr.open_array(str(tmp_path / "base.zarr"), mode="r")[:]))
    of_max = np.nanmax(np.abs(zarr.open_array(str(tmp_path / "of.zarr"), mode="r")[:]))

    # no winsorization leaves the 1e6 outlier as a large z-score; both winsorizers bound it
    assert min_max > 10.0
    assert base_max <= 6.01  # std winsorize at mean ± 6 std
    assert of_max < base_max  # percentile winsorize is tighter still


def test_fitting_does_not_leak_from_validation_rows(target_store, tmp_path):
    train_chunks, val_chunks = train_val_chunk_indices(N_CHUNKS)
    train_rows = np.concatenate(
        [np.arange(c * CHUNK_ROWS, (c + 1) * CHUNK_ROWS) for c in train_chunks]
    )

    run_prescaling(target_store, tmp_path / "clean.zarr", get_ablation("order_fix"), force=True)
    clean = zarr.open_array(str(tmp_path / "clean.zarr"), mode="r")[:]

    # corrupt only validation-chunk rows with large finite values, then re-run
    source = zarr.open_array(str(target_store), mode="r+")
    for c in val_chunks:
        source[c * CHUNK_ROWS : (c + 1) * CHUNK_ROWS, :] = 1e4
    run_prescaling(target_store, tmp_path / "dirty.zarr", get_ablation("order_fix"), force=True)
    dirty = zarr.open_array(str(tmp_path / "dirty.zarr"), mode="r")[:]

    # equal_nan: the failed-molecule row is NaN on both sides and must compare equal
    assert np.array_equal(clean[train_rows], dirty[train_rows], equal_nan=True)


def test_get_ablation_rejects_unknown_name():
    with pytest.raises(KeyError):
        get_ablation("does_not_exist")


def test_refuses_to_overwrite_without_force(target_store, tmp_path):
    out = tmp_path / "out.zarr"
    run_prescaling(target_store, out, get_ablation("minimal"), force=True)

    with pytest.raises(FileExistsError):
        run_prescaling(target_store, out, get_ablation("minimal"), force=False)
