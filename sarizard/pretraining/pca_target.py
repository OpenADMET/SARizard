"""Compress osmordred's fully-prescaled descriptor matrix with PCA to build a smaller
pretraining target, for the ``osmordred_pca80``/``pca90``/``pca95`` flavors.

Input is the ``full``-recipe prescaled osmordred store (``prescaling.py --ablation full``):
order-fixed winsorize/z-score plus correlation drop, low-variance drop, and Yeo-Johnson. PCA
is fit once on a leakage-safe sample of the train rows (the same chunk split ``prescaling.py``
and ``split.py`` derive from ``splitting.train_val_chunk_indices``), to the largest requested
explained-variance threshold; the smaller thresholds slice a prefix of the same fitted
components, since PCA orders components by descending explained variance. The retained
component scores are z-scored per column (train-row mean/std) so the masked-pretext loss
weights every retained component comparably, matching every other continuous flavor's target.

Usage:
    python pca_target.py --input-zarr cache/prescaled/osmordred_full/prescaled.zarr \\
        --output-dir cache/targets --base-flavor osmordred \\
        --thresholds 0.8 0.9 0.95 --force
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import zarr
from sklearn.decomposition import PCA

# dual import: script-style when run from pretraining/ (sbatch), package-style when imported
# from the repo root (tests)
try:
    from prescaling import DTYPE, evenly_spaced, welford_over_ranges
    from splitting import chunk_row_ranges, train_val_chunk_indices
except ImportError:
    from sarizard.pretraining.prescaling import DTYPE, evenly_spaced, welford_over_ranges
    from sarizard.pretraining.splitting import chunk_row_ranges, train_val_chunk_indices

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.80, 0.90, 0.95)
SAMPLE_FRAC = 0.3
SAMPLE_ROWS_MIN = 20_000
SAMPLE_ROWS_MAX = 100_000
PCA_SEED = 42


def threshold_flavor_name(base_flavor: str, threshold: float) -> str:
    """Return the flavor name for one PCA-threshold variant (e.g. ``osmordred_pca80``)."""
    return f"{base_flavor}_pca{round(threshold * 100)}"


def fit_pca_targets(
    input_zarr: Path,
    output_dir: Path,
    base_flavor: str,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    *,
    chunk_rows_out: int,
    force: bool = False,
) -> dict:
    """Fit PCA on the train rows of ``input_zarr`` and write one target store per threshold.

    Parameters
    ----------
    input_zarr : pathlib.Path
        The ``full``-recipe prescaled descriptor store (already cleaned, winsorized,
        decorrelated, Yeo-Johnson transformed, and z-scored).
    output_dir : pathlib.Path
        Root cache directory; each threshold's store is written to
        ``output_dir / threshold_flavor_name(base_flavor, t) / target.zarr``.
    base_flavor : str
        The flavor the input matrix was prescaled from (``"osmordred"``).
    thresholds : sequence of float, optional
        Explained-variance thresholds, each in ``(0, 1]``. Default ``DEFAULT_THRESHOLDS``.
    chunk_rows_out : int
        Row chunk size for the output stores (should match ``CORPUS_CHUNK_ROWS`` so every
        other target.zarr in the sweep shares one chunk boundary for the train/val split).
    force : bool, optional
        Overwrite existing output stores. Default ``False``.

    Returns
    -------
    dict
        Per-threshold summary: flavor name, component count, and cumulative explained
        variance actually achieved.

    Raises
    ------
    FileNotFoundError
        If ``input_zarr`` does not exist.
    """
    if not input_zarr.exists():
        raise FileNotFoundError(f"{input_zarr} not found; run the full prescale first")
    max_threshold = max(thresholds)

    arr = zarr.open_array(store=str(input_zarr), mode="r")
    n_rows, n_cols = arr.shape
    chunk_rows = arr.chunks[0]
    n_chunks = arr.nchunks

    # leakage-safe: fit PCA on the train chunks only, the same split split.py will later use
    train_chunks, _ = train_val_chunk_indices(n_chunks)
    is_train_chunk = np.zeros(n_chunks, dtype=bool)
    is_train_chunk[train_chunks] = True
    n_train_rows = int(is_train_chunk[: n_rows // chunk_rows].sum()) * chunk_rows
    sample_target = int(np.clip(n_train_rows * SAMPLE_FRAC, SAMPLE_ROWS_MIN, SAMPLE_ROWS_MAX))
    sample_idx = evenly_spaced(n_rows, sample_target)
    sample_idx = sample_idx[is_train_chunk[sample_idx // chunk_rows]]
    logger.info(
        "pca(%s): %d x %d, %d train rows, fitting on a %d-row sample, thresholds %s",
        base_flavor, n_rows, n_cols, n_train_rows, len(sample_idx), thresholds,
    )

    # gather the fit sample in one streaming pass (mirrors prescaling.py's Pass 1)
    sample = np.zeros((len(sample_idx), n_cols), dtype=DTYPE)
    sample_cursor = 0
    sample_set = set(sample_idx.tolist())
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        local = [i - start for i in range(start, end) if i in sample_set]
        if local:
            chunk = arr[start:end].astype(DTYPE)
            sample[sample_cursor : sample_cursor + len(local)] = chunk[local]
            sample_cursor += len(local)
    sample = sample[:sample_cursor]
    # PCA cannot see the masked-pretext NaN sentinel from a failed molecule; the full
    # prescaling recipe writes 0.0 for those cells already via _clean_chunk, but any residual
    # non-finite value here would poison the fit, so replace defensively
    sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)

    # one fit at the largest threshold; smaller thresholds slice a prefix of the same
    # components, since sklearn orders components by descending explained variance
    pca = PCA(n_components=max_threshold, svd_solver="full", random_state=PCA_SEED)
    pca.fit(sample)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_components_max = pca.n_components_
    logger.info(
        "pca(%s): fit %d components for %.0f%% threshold (cumulative variance %.4f)",
        base_flavor, n_components_max, max_threshold * 100, cumulative[-1],
    )

    mean_ = pca.mean_.astype(DTYPE)
    components_ = pca.components_.astype(DTYPE)  # (n_components_max, n_cols)

    def _project(chunk: np.ndarray) -> np.ndarray:
        """Project a raw chunk onto the fitted components (all n_components_max of them)."""
        clean = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
        return ((clean - mean_) @ components_.T).astype(DTYPE)

    train_ranges = chunk_row_ranges(train_chunks, chunk_rows)
    summaries = {}
    for threshold in sorted(thresholds):
        n_components = int(np.searchsorted(cumulative, threshold) + 1)
        n_components = min(n_components, n_components_max)
        flavor_name = threshold_flavor_name(base_flavor, threshold)
        out_path = output_dir / flavor_name / "target.zarr"
        if out_path.exists() and not force:
            raise FileExistsError(f"{out_path} exists; pass force=True to overwrite")

        # z-score the retained components on train rows only, matching every other
        # continuous flavor's target (masked MSE otherwise implicitly overweights the
        # higher-variance leading components)
        mean_pc, std_pc = welford_over_ranges(
            arr, train_ranges, lambda c: _project(c)[:, :n_components], n_components, chunk_rows
        )

        if out_path.exists():
            import shutil

            shutil.rmtree(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out = zarr.create_array(
            store=str(out_path),
            shape=(n_rows, n_components),
            chunks=(chunk_rows_out, n_components),
            dtype=DTYPE,
            compressors=None,
            fill_value=np.nan,
        )
        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            raw = arr[start:end].astype(DTYPE)
            scores = _project(raw)[:, :n_components]
            scaled = ((scores - mean_pc) / std_pc).astype(DTYPE)
            out[start:end] = scaled

        summaries[flavor_name] = {
            "base_flavor": base_flavor,
            "threshold": threshold,
            "n_components": n_components,
            "cumulative_explained_variance": float(cumulative[n_components - 1]),
        }
        logger.info(
            "pca(%s): threshold %.0f%% -> %s, %d components (cumulative variance %.4f), "
            "wrote %s",
            base_flavor, threshold * 100, flavor_name, n_components,
            cumulative[n_components - 1], out_path,
        )
    return summaries


def main() -> None:
    """Fit PCA on a prescaled descriptor store and write one target.zarr per threshold."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zarr", type=Path, required=True, help="full-recipe prescaled.zarr")
    parser.add_argument("--output-dir", type=Path, required=True, help="cache/targets root")
    parser.add_argument("--base-flavor", required=True, help="flavor the input was prescaled from")
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS),
        help="explained-variance thresholds in (0, 1]",
    )
    parser.add_argument("--chunk-rows-out", type=int, required=True, help="output row chunk size")
    parser.add_argument("--summary", type=Path, default=None, help="optional summary JSON path")
    parser.add_argument("--force", action="store_true", help="overwrite existing target stores")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    summary = fit_pca_targets(
        args.input_zarr, args.output_dir, args.base_flavor, args.thresholds,
        chunk_rows_out=args.chunk_rows_out, force=args.force,
    )
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
