"""Pack a flavor's cached ``target.npy`` into the chunked ``target.zarr`` the trainer reads.

Run in the main ``sarizard`` environment (zarr 3.x). For direct flavors that already run
in ``sarizard`` this immediately follows ``compute_target``; for the isolated-env flavors it
is the second step after their calculator has written the ``.npy``.

Usage:
    python -m pretraining.features.pack_target --flavor ecfp
"""

from __future__ import annotations

import argparse
import logging

from analysis.paths import target_npy, target_zarr
from pretraining.config import CORPUS_CHUNK_ROWS
from pretraining.features._pack import pack_npy_to_zarr
from pretraining.flavors import get_flavor

logger = logging.getLogger(__name__)


def main() -> None:
    """Pack one flavor's ``target.npy`` into ``target.zarr``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavor", required=True, help="flavor name (see flavors.py)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing target.zarr")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    flavor = get_flavor(args.flavor)
    pack_npy_to_zarr(
        target_npy(flavor.name), target_zarr(flavor.name), CORPUS_CHUNK_ROWS, force=args.force
    )


if __name__ == "__main__":
    main()
