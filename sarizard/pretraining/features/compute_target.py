"""Compute one flavor's pretraining target and cache it as ``target.npy``.

Run inside the flavor's conda environment (``FLAVORS[name].env``). Reads the shared corpus
SMILES, dispatches to the flavor's calculator, and writes ``cache/targets/<flavor>/target.npy``
(numpy only, so the isolated old-Python envs need no zarr). Then run ``pack_target`` in the
main ``sarizard`` environment to produce the chunked ``target.zarr`` the trainer reads.

The ``surrogate_adme`` flavor is an exception: it reads the Novartis released CSV directly
(``--csv-path``) and writes both ``target.npy`` and a companion ``corpus_smiles.parquet``
that replaces the shared corpus for that flavor's pretrain step.

Usage:
    python -m sarizard.pretraining.features.compute_target --flavor ecfp
    python -m sarizard.pretraining.features.compute_target --flavor osmordred --n-jobs 32
    python -m sarizard.pretraining.features.compute_target --flavor surrogate_adme \\
        --csv-path /data/protacdb2.0_zinc_chembl_dataset.csv
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from sarizard.analysis.paths import CORPUS_SMILES, target_npy, target_shard
from sarizard.pretraining.config import COMPUTE_BLOCK_ROWS
from sarizard.pretraining.features import _npy
from sarizard.pretraining.flavors import Flavor, get_flavor

logger = logging.getLogger(__name__)


def _read_smiles(corpus: Path) -> list[str]:
    """Read the SMILES column from the corpus parquet (pyarrow keeps env deps light)."""
    return pq.read_table(corpus, columns=["SMILES"]).column("SMILES").to_pylist()


def _streaming_compute_fn(
    flavor: Flavor, n_jobs: int, batch_size: int
) -> Callable[[Sequence[str]], np.ndarray] | None:
    """Return the streaming calculator for a flavor, or None if it produces a whole array.

    Each branch imports its calculator lazily so a conflicting or heavy dependency only
    loads in the environment that actually computes that flavor.
    """
    from sarizard.pretraining.features.skfp_targets import is_skfp_flavor

    # a flavor may reuse another's calculator on a different corpus (e.g. osmordred_surrogate
    # runs the osmordred calculator on the Novartis molecules), so dispatch on calculator
    name = flavor.calculator or flavor.name
    if is_skfp_flavor(name):
        from sarizard.pretraining.features.skfp_targets import build_compute_fn

        return build_compute_fn(name, n_jobs, flavor.target_dim)
    if name == "osmordred":
        from sarizard.pretraining.features.osmordred_target import build_compute_fn

        return build_compute_fn(n_jobs)
    if name == "minimol":
        from sarizard.pretraining.features.minimol_target import build_compute_fn

        return build_compute_fn(batch_size)
    if name == "jazzy":
        from sarizard.pretraining.features.jazzy_target import build_compute_fn

        return build_compute_fn(n_jobs)
    return None


# rows copied per block when merging shards, so a wide shard never loads whole into memory
_MERGE_BLOCK = 8192


def _shard_bounds(n_rows: int, num_shards: int, shard_index: int) -> tuple[int, int]:
    """Return the ``[start, end)`` row range this shard owns of a contiguous split.

    The shards tile ``range(n_rows)`` with no gaps or overlap, so concatenating them in
    index order reproduces the full corpus order the cache must preserve.
    """
    if not 0 <= shard_index < num_shards:
        raise SystemExit(f"shard-index {shard_index} out of range for {num_shards} shards")
    per_shard = -(-n_rows // num_shards)  # ceil so the last shard absorbs the remainder
    start = shard_index * per_shard
    end = min(start + per_shard, n_rows)
    if start >= n_rows:
        raise SystemExit(
            f"shard {shard_index}/{num_shards} is empty for a {n_rows}-row corpus; "
            "use fewer shards"
        )
    return start, end


def _merge_shards(paths: list[Path], out: Path, expected_rows: int, target_dim: int) -> None:
    """Concatenate row-shards into ``out`` in order, validating completeness first.

    Reads only ``.npy`` headers and copies block by block, so merging never materializes a
    whole shard. Raises if a shard is missing, mis-shaped, or the totals do not match the
    corpus, so an incomplete sharded run cannot be packed as a finished target.
    """
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"cannot merge: {len(missing)} shard(s) missing, e.g. {missing[:3]}")
    shapes = [np.load(p, mmap_mode="r").shape for p in paths]
    if any(len(s) != 2 or s[1] != target_dim for s in shapes):
        raise SystemExit(f"shard width mismatch (expected dim {target_dim}): {shapes}")
    total = sum(s[0] for s in shapes)
    if total != expected_rows:
        raise SystemExit(f"merged rows {total} != corpus rows {expected_rows}; shards incomplete")

    memmap = _npy.open_target_memmap(out, total, target_dim)
    row = 0
    for path, shape in zip(paths, shapes, strict=True):
        shard = np.load(path, mmap_mode="r")
        for start in range(0, shape[0], _MERGE_BLOCK):
            end = min(start + _MERGE_BLOCK, shape[0])
            memmap[row + start : row + end] = shard[start:end]
        row += shape[0]
        memmap.flush()
        del shard
    logger.info("merged %d shards -> %s (%d rows, %d dim)", len(paths), out, total, target_dim)


def main() -> None:
    """Compute and cache one flavor's target as ``target.npy``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavor", required=True, help="flavor name (see flavors.py)")
    parser.add_argument("--corpus", type=Path, default=CORPUS_SMILES, help="corpus SMILES parquet")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="surrogate_adme only: path to protacdb2.0_zinc_chembl_dataset.csv",
    )
    parser.add_argument("--out", type=Path, default=None, help="output .npy (default cache path)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel jobs for calculators")
    parser.add_argument("--batch-size", type=int, default=100, help="minimol internal batch size")
    parser.add_argument("--block-rows", type=int, default=COMPUTE_BLOCK_ROWS, help="rows/block")
    parser.add_argument("--force", action="store_true", help="overwrite an existing .npy")
    parser.add_argument(
        "--num-shards", type=int, default=1, help="split the corpus into this many row-shards"
    )
    parser.add_argument(
        "--shard-index", type=int, default=None, help="which shard this task computes (0-based)"
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="merge previously computed shards into target.npy (no calculator; main env)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    flavor = get_flavor(args.flavor)
    if flavor.target_dim is None:
        raise SystemExit(f"flavor {flavor.name} has no known target_dim; set it in flavors.py")

    # merge mode runs after a sharded array job, in the main env: concatenate the shards into
    # target.npy (no calculator needed), which pack_target then turns into the zarr
    if args.merge_shards:
        shard_paths = [
            target_shard(flavor.name, k, args.num_shards) for k in range(args.num_shards)
        ]
        expected_rows = pq.read_metadata(args.corpus).num_rows
        merged_out = args.out or target_npy(flavor.name)
        _merge_shards(shard_paths, merged_out, expected_rows, flavor.target_dim)
        return

    # a sharded task writes only its row slice to a shard file; an unsharded run writes target.npy
    sharded = args.num_shards > 1
    if sharded:
        if args.shard_index is None:
            raise SystemExit("--num-shards > 1 requires --shard-index")
        if flavor.name == "surrogate_adme":
            raise SystemExit("surrogate_adme has its own corpus and cannot be sharded")
        out = target_shard(flavor.name, args.shard_index, args.num_shards)
    else:
        out = args.out or target_npy(flavor.name)
    if out.exists() and not args.force:
        # a resumable skip is a success, not a failure: exiting nonzero would break the afterok
        # dependency chain in the slurm pipeline. but only skip a real target, not a crash
        # orphan: a prior run that died mid-block leaves an all-NaN memmap on disk, and blindly
        # skipping would pack that garbage. sample ~1000 rows to tell a real target from an orphan
        existing = np.load(out, mmap_mode="r")
        stride = max(1, existing.shape[0] // 1000)
        has_data = bool(np.isfinite(np.asarray(existing[::stride])).any())
        del existing
        if has_data:
            logger.info("%s exists; skipping (pass --force to overwrite)", out)
            return
        logger.warning("%s exists but is entirely NaN (crash orphan); recomputing", out)
        out.unlink()

    active_env = os.environ.get("CONDA_DEFAULT_ENV")
    if active_env and active_env != flavor.env:
        logger.warning(
            "active env %r != flavor env %r; calculator deps may be missing",
            active_env,
            flavor.env,
        )

    # surrogate_adme reads its own corpus from the released CSV rather than the shared corpus
    if flavor.name == "surrogate_adme":
        if args.csv_path is None:
            raise SystemExit(
                "surrogate_adme requires --csv-path <protacdb2.0_zinc_chembl_dataset.csv>"
            )
        from sarizard.pretraining.features.surrogate_target import build_from_csv

        n_kept = build_from_csv(args.csv_path, out, force=args.force)
        logger.info(
            "wrote surrogate target (%d rows); also wrote corpus_smiles.parquet for split.py",
            n_kept,
        )
        return

    smiles = _read_smiles(args.corpus)
    if sharded:
        start, end = _shard_bounds(len(smiles), args.num_shards, args.shard_index)
        logger.info(
            "shard %d/%d: rows [%d, %d) of %d",
            args.shard_index,
            args.num_shards,
            start,
            end,
            len(smiles),
        )
        smiles = smiles[start:end]
    logger.info(
        "flavor=%s kind=%s dim=%d rows=%d",
        flavor.name,
        flavor.kind,
        flavor.target_dim,
        len(smiles),
    )
    memmap = _npy.open_target_memmap(out, len(smiles), flavor.target_dim)

    compute_fn = _streaming_compute_fn(flavor, args.n_jobs, args.batch_size)
    if compute_fn is not None:
        n_failed = _npy.fill_streaming(memmap, smiles, compute_fn, args.block_rows)
    else:
        raise SystemExit(f"no calculator registered for flavor {flavor.name}")

    # every row all-NaN means the calculator never produced a value (wrong env, broken
    # extension, unreadable corpus); caching that garbage would silently poison prescaling
    # downstream, so delete the useless output and fail loudly here at the source
    if n_failed == len(smiles):
        del memmap
        out.unlink(missing_ok=True)
        raise SystemExit(
            f"flavor {flavor.name}: all {len(smiles)} rows failed (target is entirely NaN); "
            "check the calculator environment, not a resumable skip"
        )

    logger.info(
        "wrote %s (%d rows, %d failed); now pack with sarizard.pretraining.features.pack_target",
        out,
        len(smiles),
        n_failed,
    )


if __name__ == "__main__":
    main()
