"""Generate per-flavor finetuning recipes from the committed baseline templates.

For every flavor in the registry and every baseline (dataset, endpoint) recipe in
``configs/_baseline/``, emit ``configs/<flavor>/<endpoint>.yaml`` identical to the baseline
except that the ChemProp model initializes from that flavor's converted foundation
checkpoint instead of stock CheMeleon. The finetuning regime is held fixed across flavors;
only the foundation differs, so the report-card columns stay comparable.

The baseline templates are the stock-CheMeleon recipes copied from the sibling igm project.
Running them unchanged yields a stock-CheMeleon reference column (a different corpus and
regime than our flavors, so a reference rather than an apples-to-apples arm).

Usage:
    python -m configs.generate                      # all flavors, all endpoints
    python -m configs.generate --flavors ecfp jazzy # a subset of flavors
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from analysis.paths import CONFIGS_DIR, REPO_ROOT, foundation_path
from pretraining.flavors import flavor_names

logger = logging.getLogger(__name__)

BACKBONE = "chemeleon"  # token the baseline recipes use in names, tags, and from_foundation
BASELINE_DIR = CONFIGS_DIR / "_baseline"


def _retarget(recipe: dict, flavor: str, accelerator: str) -> dict:
    """Point a baseline recipe at a flavor's foundation and relabel it for provenance."""
    # initialize from this flavor's converted foundation (repo-relative; anvil runs from root)
    params = recipe["procedure"]["model"]["params"]
    params["from_foundation"] = str(foundation_path(flavor).relative_to(REPO_ROOT))

    # normalize the accelerator: baselines pin a laptop device, cluster nodes resolve "auto"
    train_params = recipe.get("procedure", {}).get("train", {}).get("params")
    if isinstance(train_params, dict) and "accelerator" in train_params:
        train_params["accelerator"] = accelerator

    # relabel machine identifiers from the backbone to the flavor
    meta = recipe.get("metadata", {})
    for key in ("name", "tag"):
        if isinstance(meta.get(key), str):
            meta[key] = meta[key].replace(BACKBONE, flavor)
    if isinstance(meta.get("tags"), list):
        meta["tags"] = [flavor if tag == BACKBONE else tag for tag in meta["tags"]]
    return recipe


def _endpoint_name(template: Path) -> str:
    """Strip the backbone prefix so files sit under ``configs/<flavor>/<endpoint>.yaml``."""
    stem = template.stem
    prefix = f"{BACKBONE}_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def main() -> None:
    """Generate per-flavor recipes for the requested flavors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavors", nargs="*", default=None, help="flavor subset (default all)")
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR, help="templates dir")
    parser.add_argument("--accelerator", default="auto", help="lightning accelerator")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    templates = sorted(args.baseline_dir.glob(f"{BACKBONE}_*.yaml"))
    if not templates:
        raise SystemExit(f"no baseline templates in {args.baseline_dir}")
    flavors = args.flavors or flavor_names()

    n_written = 0
    for flavor in flavors:
        out_dir = CONFIGS_DIR / flavor
        out_dir.mkdir(parents=True, exist_ok=True)
        for template in templates:
            recipe = yaml.safe_load(template.read_text())
            recipe = _retarget(recipe, flavor, args.accelerator)
            (out_dir / f"{_endpoint_name(template)}.yaml").write_text(
                yaml.safe_dump(recipe, sort_keys=False)
            )
            n_written += 1
        logger.info("flavor %s: %d recipes", flavor, len(templates))
    logger.info("wrote %d recipes across %d flavors", n_written, len(flavors))


if __name__ == "__main__":
    main()
