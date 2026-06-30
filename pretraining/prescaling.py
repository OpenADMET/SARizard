"""Toggleable descriptor prescaling for the continuous pretraining targets.

Adapted from the OpenADMET foundation-models ``preprocessing/prescaling.py`` reference,
restricted to local zarr stores and reworked so each step can be switched on or off
independently. This makes the pipeline an ablation instrument: run the same osmordred
target through several :class:`PrescalingConfig` variants, pretrain and finetune from each,
and read which preprocessing steps actually help downstream before baking one recipe into
the flavor sweep.

The full pipeline, in canonical order, is:

1. drop invalid (mandatory): drop columns whose NaN/inf fraction exceeds a threshold, then
   replace remaining NaN with 0, +inf with the column max, and -inf with the column min.
2. winsorize: clip each column to a robust range (percentile- or std-based).
3. drop correlated: drop the later column of each pair with sampled ``|r| > threshold``.
4. Yeo-Johnson: fit and apply a per-column power transform toward normality.
5. drop low variance: drop columns whose post-transform variance is ~0.
6. z-score: subtract the mean and divide by the std.

All fitting (percentiles, correlation, Yeo-Johnson lambdas, variance, mean/std) happens on
the train chunks only, drawn from the same chunk split ``split.py`` later emits, so nothing
from the validation rows leaks into the transform. Column drops and transforms are then
applied to every row when writing the output store.

Reproducing the current CheMeleon recipe (``chemeleon_baseline`` ablation): the production
``split.py`` computes mean/std on the raw target and uses those same stats both to set the
winsorization limits (mean ± k·std) and to z-score, so the outliers it is about to clip
inflate the std first. That is ``winsorize_method="std"`` with ``zscore_fit="raw"``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import zarr
from scipy.stats import yeojohnson

# dual import: script-style when run from pretraining/ (sbatch), package-style when imported
# from the repo root (analysis package, slurm env.sh helper)
try:
    from splitting import chunk_row_ranges, train_val_chunk_indices
except ImportError:
    from pretraining.splitting import chunk_row_ranges, train_val_chunk_indices

logger = logging.getLogger(__name__)

DTYPE = np.float32


@dataclass(frozen=True)
class PrescalingConfig:
    """One descriptor-prescaling recipe: which steps run and with what parameters.

    Parameters
    ----------
    name : str
        Recipe key, used for cache and output directory names.
    do_drop_invalid : bool, optional
        Drop columns whose NaN/inf fraction exceeds ``invalid_frac_threshold``. The NaN/inf
        value replacement is always applied regardless; only the column drop is gated.
    invalid_frac_threshold : float, optional
        Maximum NaN/inf fraction a column may have and still be kept.
    do_winsorize : bool, optional
        Clip each column to a robust range before scaling.
    winsorize_method : {"percentile", "std"}, optional
        ``percentile`` clips to ``winsorize_limits`` quantiles; ``std`` clips to
        ``mean ± winsorize_std_factor · std`` (the CheMeleon style).
    winsorize_limits : tuple of float, optional
        Lower and upper tail fractions for the percentile method.
    winsorize_std_factor : float, optional
        Standard-deviation multiplier for the std method.
    do_drop_corr : bool, optional
        Drop the later column of each highly correlated pair.
    corr_threshold : float, optional
        Absolute Pearson correlation above which a pair is considered redundant.
    do_yeo_johnson : bool, optional
        Fit and apply a per-column Yeo-Johnson power transform.
    do_drop_low_var : bool, optional
        Drop columns whose post-transform variance is below ``low_var_threshold``.
    low_var_threshold : float, optional
        Variance floor for the low-variance drop.
    do_zscore : bool, optional
        Subtract the mean and divide by the std.
    zscore_fit : {"post_transform", "raw"}, optional
        ``post_transform`` fits mean/std on the winsorized and power-transformed data;
        ``raw`` fits on the cleaned-but-untransformed data (the CheMeleon style).
    sample_frac, sample_rows_min, sample_rows_max : optional
        Control the size of the train-row sample used to fit percentiles, correlation,
        Yeo-Johnson lambdas, and variance.
    corr_sample_rows_max : int, optional
        Row cap for the correlation estimate (a Gram matrix over the sample).
    description : str, optional
        Short human-readable summary for the wiki and logs.
    """

    name: str
    do_drop_invalid: bool = True
    invalid_frac_threshold: float = 0.2
    do_winsorize: bool = True
    winsorize_method: str = "percentile"
    winsorize_limits: tuple[float, float] = (0.01, 0.01)
    winsorize_std_factor: float = 6.0
    do_drop_corr: bool = False
    corr_threshold: float = 0.98
    do_yeo_johnson: bool = False
    do_drop_low_var: bool = False
    low_var_threshold: float = 1e-8
    do_zscore: bool = True
    zscore_fit: str = "post_transform"
    sample_frac: float = 0.3
    sample_rows_min: int = 20_000
    sample_rows_max: int = 100_000
    corr_sample_rows_max: int = 20_000
    description: str = ""


# The ablation ladder, run before the flavor sweep to cement the production recipe.
# chemeleon_baseline reproduces today's split.py; order_fix corrects the winsorize/scale
# entanglement; each plus_* isolates one new step on top of order_fix; full stacks them.
_ORDER_FIX = PrescalingConfig(
    name="order_fix",
    winsorize_method="percentile",
    zscore_fit="post_transform",
    description="Correct order: winsorize (percentile) first, then z-score on winsorized data.",
)

ABLATIONS: dict[str, PrescalingConfig] = {
    "minimal": PrescalingConfig(
        name="minimal",
        do_winsorize=False,
        zscore_fit="post_transform",
        description="Floor: mandatory NaN/inf clean and z-score only, no winsorization.",
    ),
    "chemeleon_baseline": PrescalingConfig(
        name="chemeleon_baseline",
        do_winsorize=True,
        winsorize_method="std",
        winsorize_std_factor=6.0,
        zscore_fit="raw",
        description=(
            "Reproduces production split.py: std-based winsorize and z-score both fit on "
            "the raw target, so clipped outliers inflate the std first."
        ),
    ),
    "order_fix": _ORDER_FIX,
    "plus_drop_corr": replace(
        _ORDER_FIX,
        name="plus_drop_corr",
        do_drop_corr=True,
        description="order_fix plus dropping one column of each |r| > 0.98 pair.",
    ),
    "plus_drop_low_var": replace(
        _ORDER_FIX,
        name="plus_drop_low_var",
        do_drop_low_var=True,
        description="order_fix plus dropping near-zero-variance columns.",
    ),
    "plus_yeo_johnson": replace(
        _ORDER_FIX,
        name="plus_yeo_johnson",
        do_yeo_johnson=True,
        description="order_fix plus a per-column Yeo-Johnson power transform.",
    ),
    "full": replace(
        _ORDER_FIX,
        name="full",
        do_drop_corr=True,
        do_yeo_johnson=True,
        do_drop_low_var=True,
        description="All steps stacked in canonical order.",
    ),
}


def get_ablation(name: str) -> PrescalingConfig:
    """Return the prescaling config registered under ``name``."""
    try:
        return ABLATIONS[name]
    except KeyError as err:
        raise KeyError(
            f"unknown ablation {name!r}; known: {', '.join(ABLATIONS)}"
        ) from err


def ablation_names() -> list[str]:
    """Return all registered ablation names in definition order."""
    return list(ABLATIONS)


def _clean_chunk(
    chunk: np.ndarray,
    col_max: np.ndarray,
    col_min: np.ndarray,
    keep_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Drop filtered columns, then replace NaN with 0, +inf with col_max, -inf with col_min."""
    if keep_mask is not None:
        chunk = chunk[:, keep_mask]
        col_max = col_max[keep_mask]
        col_min = col_min[keep_mask]
    chunk = np.where(np.isnan(chunk), 0.0, chunk)
    chunk = np.where(np.isposinf(chunk), col_max, chunk)
    chunk = np.where(np.isneginf(chunk), col_min, chunk)
    return chunk.astype(DTYPE)


def _fit_yj_cols(sample: np.ndarray, col_indices: np.ndarray, lambdas_out: list) -> None:
    """Fit a Yeo-Johnson lambda for each column index, storing into ``lambdas_out``."""
    for j in col_indices:
        col = sample[:, j].astype(np.float64)
        try:
            _, lmbda = yeojohnson(col)
            lambdas_out[j] = float(lmbda)
        except (ValueError, RuntimeError):
            # a degenerate column (e.g. constant) leaves itself untransformed
            lambdas_out[j] = None


def _apply_yj_cols(
    chunk: np.ndarray, out: np.ndarray, lambdas: list, col_indices: np.ndarray
) -> None:
    """Apply pre-fit Yeo-Johnson lambdas to selected columns of ``chunk`` into ``out``."""
    for k in col_indices:
        lmbda = lambdas[k]
        col = chunk[:, k].astype(np.float64)
        if lmbda is None:
            out[:, k] = col.astype(DTYPE)
            continue
        try:
            out[:, k] = np.asarray(yeojohnson(col, lmbda=lmbda), dtype=DTYPE)
        except (ValueError, RuntimeError):
            logger.warning("Yeo-Johnson failed for column %d; leaving untransformed", k)
            out[:, k] = col.astype(DTYPE)


def _evenly_spaced(n: int, max_n: int) -> np.ndarray:
    """Return up to ``max_n`` evenly spaced indices from ``range(n)``."""
    if n <= max_n:
        return np.arange(n)
    step = max(1, n // max_n)
    return np.arange(0, n, step)[:max_n]


def _drop_high_corr_columns(
    sample: np.ndarray, threshold: float, max_rows: int, block_rows: int = 4096
) -> np.ndarray:
    """Return a keep-mask dropping the later column of each highly correlated pair."""
    rows = _evenly_spaced(sample.shape[0], max_rows)
    n = len(rows)
    n_cols = sample.shape[1]
    if n < 2:
        return np.ones(n_cols, dtype=bool)

    # column mean and std over the sampled rows, accumulated in blocks
    sums = np.zeros(n_cols, dtype=np.float64)
    sq_sums = np.zeros(n_cols, dtype=np.float64)
    for start in range(0, n, block_rows):
        block = sample[rows[start : start + block_rows]].astype(np.float64, copy=False)
        sums += block.sum(axis=0)
        sq_sums += np.square(block).sum(axis=0)
    mean = sums / n
    var = (sq_sums - n * np.square(mean)) / (n - 1)
    std = np.sqrt(np.where(var > 1e-12, var, 1.0))

    # standardized Gram matrix -> correlation matrix
    gram = np.zeros((n_cols, n_cols), dtype=np.float64)
    for start in range(0, n, block_rows):
        block = sample[rows[start : start + block_rows]].astype(np.float64, copy=False)
        block = (block - mean) / std
        gram += block.T @ block
    corr = np.clip(gram / (n - 1), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    # for each correlated pair still both kept, drop the later column
    pair_i, pair_j = np.where(np.triu(np.abs(corr) > threshold, k=1))
    keep_mask = np.ones(n_cols, dtype=bool)
    for i, j in zip(pair_i.tolist(), pair_j.tolist(), strict=True):
        if keep_mask[i] and keep_mask[j]:
            keep_mask[j] = False
    return keep_mask


def _welford_over_ranges(
    arr: zarr.Array,
    ranges: Sequence[tuple[int, int]],
    transform: Callable[[np.ndarray], np.ndarray],
    n_cols: int,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact mean/std over given row ranges after applying ``transform``.

    Streams each range in ``chunk_rows`` blocks, accumulating Welford statistics so the
    full dataset never loads into memory. Returns ``(mean, std)`` as float32, with std
    floored to 1.0 where a column has zero variance.
    """
    running_n = 0
    running_mean = np.zeros(n_cols, dtype=np.float64)
    running_m2 = np.zeros(n_cols, dtype=np.float64)
    for lo, hi in ranges:
        for start in range(lo, hi, chunk_rows):
            end = min(start + chunk_rows, hi)
            block = transform(arr[start:end].astype(DTYPE)).astype(np.float64)
            block_n = block.shape[0]
            if block_n == 0:
                continue
            block_mean = block.mean(axis=0)
            block_m2 = np.square(block - block_mean).sum(axis=0)
            delta = block_mean - running_mean
            new_n = running_n + block_n
            running_mean += delta * (block_n / new_n)
            running_m2 += block_m2 + delta**2 * (running_n * block_n / new_n)
            running_n = new_n
    if running_n < 2:
        raise ValueError(f"need >= 2 rows to compute variance, got {running_n}")
    mean = running_mean.astype(DTYPE)
    var = running_m2 / (running_n - 1)
    std = np.where(var > 0, np.sqrt(var), 1.0).astype(DTYPE)
    return mean, std


def run_prescaling(
    input_zarr: Path, output_zarr: Path, cfg: PrescalingConfig, *, force: bool = False
) -> dict:
    """Prescale a raw target store into a new store under one config, leakage-safe.

    Parameters
    ----------
    input_zarr : pathlib.Path
        Raw per-flavor ``target.zarr`` (rows are molecules, columns descriptor targets).
    output_zarr : pathlib.Path
        Destination ``prescaled.zarr`` store.
    cfg : PrescalingConfig
        Which steps to run and with what parameters.
    force : bool, optional
        Overwrite an existing destination store. Default ``False``.

    Returns
    -------
    dict
        Summary of the run: input/output column counts and the per-step columns dropped.

    Raises
    ------
    FileNotFoundError
        If ``input_zarr`` does not exist.
    FileExistsError
        If ``output_zarr`` exists and ``force`` is ``False``.
    """
    if not input_zarr.exists():
        raise FileNotFoundError(f"{input_zarr} not found; compute and pack the target first")
    if output_zarr.exists() and not force:
        raise FileExistsError(f"{output_zarr} exists; pass force=True to overwrite")

    arr = zarr.open_array(store=str(input_zarr), mode="r")
    n_rows, n_cols = arr.shape
    if n_rows == 0:
        raise ValueError("input array is empty")
    chunk_rows = arr.chunks[0]
    n_chunks = arr.nchunks
    max_workers = max(1, (os.cpu_count() or 2) - 1)

    # leakage-safe: fit on the train chunks split.py will later emit
    train_chunks, _ = train_val_chunk_indices(n_chunks)
    is_train_chunk = np.zeros(n_chunks, dtype=bool)
    is_train_chunk[train_chunks] = True
    n_train_rows = int(is_train_chunk[: n_rows // chunk_rows].sum()) * chunk_rows
    sample_target = int(
        np.clip(n_train_rows * cfg.sample_frac, cfg.sample_rows_min, cfg.sample_rows_max)
    )
    sample_idx = _evenly_spaced(n_rows, sample_target)
    sample_idx = sample_idx[is_train_chunk[sample_idx // chunk_rows]]
    logger.info(
        "prescaling %s: %d x %d, %d train rows, sample %d, steps "
        "[invalid=%s winsor=%s(%s) corr=%s yj=%s lowvar=%s zscore=%s(%s)]",
        cfg.name, n_rows, n_cols, n_train_rows, len(sample_idx),
        cfg.do_drop_invalid, cfg.do_winsorize, cfg.winsorize_method, cfg.do_drop_corr,
        cfg.do_yeo_johnson, cfg.do_drop_low_var, cfg.do_zscore, cfg.zscore_fit,
    )

    # ── Pass 1: column ranges, invalid counts, and the train-row sample ───────────
    col_max = np.full(n_cols, -np.inf, dtype=np.float64)
    col_min = np.full(n_cols, np.inf, dtype=np.float64)
    invalid_counts = np.zeros(n_cols, dtype=np.int64)
    sample = np.zeros((len(sample_idx), n_cols), dtype=DTYPE)
    sample_cursor = 0
    sample_set = set(sample_idx.tolist())
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        chunk = arr[start:end].astype(DTYPE)
        finite = np.isfinite(chunk)
        invalid_counts += (~finite).sum(axis=0)
        col_max = np.maximum(col_max, np.where(finite, chunk, -np.inf).max(axis=0))
        col_min = np.minimum(col_min, np.where(finite, chunk, np.inf).min(axis=0))
        local = [i - start for i in range(start, end) if i in sample_set]
        if local:
            sample[sample_cursor : sample_cursor + len(local)] = chunk[local]
            sample_cursor += len(local)
    sample = sample[:sample_cursor]
    col_max_f = col_max.astype(DTYPE)
    col_min_f = col_min.astype(DTYPE)
    has_finite = col_max > -np.inf

    # ── Step: drop columns with too many NaN/inf (mandatory cleaning of values) ───
    if cfg.do_drop_invalid:
        invalid_frac = invalid_counts / n_rows
        valid_mask = (invalid_frac <= cfg.invalid_frac_threshold) & has_finite
    else:
        valid_mask = has_finite
    if not valid_mask.any():
        raise ValueError("no columns remain after the invalid-column drop")
    valid_indices = np.where(valid_mask)[0]
    dropped_invalid = np.where(~valid_mask)[0]
    sample = _clean_chunk(sample, col_max_f, col_min_f, valid_mask)
    n_valid = sample.shape[1]

    # ── raw train mean/std, needed for std-winsorize and/or raw z-scoring ─────────
    raw_mean = raw_std = None
    if cfg.winsorize_method == "std" or cfg.zscore_fit == "raw":
        train_ranges = chunk_row_ranges(train_chunks, chunk_rows)
        raw_mean, raw_std = _welford_over_ranges(
            arr,
            train_ranges,
            lambda c: _clean_chunk(c, col_max_f, col_min_f, valid_mask),
            n_valid,
            chunk_rows,
        )

    # ── Step: winsorize ──────────────────────────────────────────────────────────
    if cfg.do_winsorize and cfg.winsorize_method == "percentile":
        lo_frac, hi_frac = cfg.winsorize_limits
        lower = np.percentile(sample, lo_frac * 100, axis=0).astype(DTYPE)
        upper = np.percentile(sample, (1 - hi_frac) * 100, axis=0).astype(DTYPE)
    elif cfg.do_winsorize and cfg.winsorize_method == "std":
        lower = (raw_mean - cfg.winsorize_std_factor * raw_std).astype(DTYPE)
        upper = (raw_mean + cfg.winsorize_std_factor * raw_std).astype(DTYPE)
    elif cfg.do_winsorize:
        raise ValueError(f"unknown winsorize_method {cfg.winsorize_method!r}")
    else:
        lower = np.full(n_valid, -np.inf, dtype=DTYPE)
        upper = np.full(n_valid, np.inf, dtype=DTYPE)
    sample = np.clip(sample, lower, upper).astype(DTYPE)

    # ── Step: drop highly correlated columns ─────────────────────────────────────
    if cfg.do_drop_corr:
        corr_keep = _drop_high_corr_columns(sample, cfg.corr_threshold, cfg.corr_sample_rows_max)
    else:
        corr_keep = np.ones(n_valid, dtype=bool)
    dropped_corr = valid_indices[~corr_keep]
    sample = sample[:, corr_keep]
    lower, upper = lower[corr_keep], upper[corr_keep]
    valid_indices = valid_indices[corr_keep]
    if raw_mean is not None:
        raw_mean, raw_std = raw_mean[corr_keep], raw_std[corr_keep]
    n_valid = sample.shape[1]

    # ── Step: Yeo-Johnson ────────────────────────────────────────────────────────
    if cfg.do_yeo_johnson:
        yj_lambdas: list = [None] * n_valid
        splits = np.array_split(np.arange(n_valid), max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(lambda cols: _fit_yj_cols(sample, cols, yj_lambdas), splits))
        yj_sample = np.empty_like(sample)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(lambda cols: _apply_yj_cols(sample, yj_sample, yj_lambdas, cols), splits))
        sample = yj_sample
    else:
        yj_lambdas = [None] * n_valid

    # ── Step: drop low-variance columns ──────────────────────────────────────────
    if cfg.do_drop_low_var:
        var_keep = sample.var(axis=0) > cfg.low_var_threshold
    else:
        var_keep = np.ones(n_valid, dtype=bool)
    if not var_keep.any():
        raise ValueError("no columns remain after the low-variance drop")
    dropped_low_var = valid_indices[~var_keep]
    valid_indices = valid_indices[var_keep]
    kept_lower, kept_upper = lower[var_keep], upper[var_keep]
    kept_lambdas = [yj_lambdas[j] for j in np.where(var_keep)[0]]
    n_kept = int(var_keep.sum())
    if raw_mean is not None:
        raw_mean, raw_std = raw_mean[var_keep], raw_std[var_keep]

    # the column transform applied to every output chunk (clean -> drop -> clip -> YJ)
    def _transform(chunk: np.ndarray) -> np.ndarray:
        out = _clean_chunk(chunk, col_max_f, col_min_f, valid_mask)
        out = out[:, corr_keep][:, var_keep]
        out = np.clip(out, kept_lower, kept_upper).astype(DTYPE)
        if cfg.do_yeo_johnson:
            yj = np.empty_like(out)
            cols = np.arange(n_kept)
            _apply_yj_cols(out, yj, kept_lambdas, cols)
            out = yj
        return out

    # ── z-score statistics (train rows only) ─────────────────────────────────────
    if not cfg.do_zscore:
        mean, std = np.zeros(n_kept, dtype=DTYPE), np.ones(n_kept, dtype=DTYPE)
    elif cfg.zscore_fit == "raw":
        mean, std = raw_mean, raw_std
    elif cfg.zscore_fit == "post_transform":
        train_ranges = chunk_row_ranges(train_chunks, chunk_rows)
        mean, std = _welford_over_ranges(arr, train_ranges, _transform, n_kept, chunk_rows)
    else:
        raise ValueError(f"unknown zscore_fit {cfg.zscore_fit!r}")

    # ── Pass: write the transformed, scaled output store for every row ───────────
    if output_zarr.exists():
        import shutil

        shutil.rmtree(output_zarr)
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    out = zarr.create_array(
        store=str(output_zarr),
        shape=(n_rows, n_kept),
        chunks=(chunk_rows, n_kept),
        dtype=DTYPE,
        compressors=None,
        fill_value=np.nan,
    )
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        transformed = _transform(arr[start:end].astype(DTYPE))
        out[start:end] = ((transformed - mean) / std).astype(DTYPE)

    summary = {
        "name": cfg.name,
        "n_cols_in": int(n_cols),
        "n_cols_out": int(n_kept),
        "dropped_invalid": dropped_invalid.tolist(),
        "dropped_corr": dropped_corr.tolist(),
        "dropped_low_var": dropped_low_var.tolist(),
    }
    logger.info(
        "prescaled %s -> %s: %d -> %d cols (invalid -%d, corr -%d, lowvar -%d)",
        input_zarr.name, output_zarr, n_cols, n_kept,
        len(dropped_invalid), len(dropped_corr), len(dropped_low_var),
    )
    return summary


def main() -> None:
    """Prescale a flavor's target store under one ablation config."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zarr", type=Path, required=True, help="raw target.zarr")
    parser.add_argument("--output-zarr", type=Path, required=True, help="prescaled.zarr out")
    parser.add_argument("--ablation", required=True, help="ablation name (see ABLATIONS)")
    parser.add_argument("--summary", type=Path, default=None, help="optional summary JSON path")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = get_ablation(args.ablation)
    summary = run_prescaling(args.input_zarr, args.output_zarr, cfg, force=args.force)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
