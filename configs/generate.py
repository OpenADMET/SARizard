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


def _retarget(recipe: dict, foundation_rel: str, label: str, accelerator: str) -> dict:
    """Point a baseline recipe at a foundation and relabel it for provenance.

    Parameters
    ----------
    recipe : dict
        A parsed baseline recipe (mutated in place and returned).
    foundation_rel : str
        Repo-relative path to the foundation checkpoint for ``from_foundation``.
    label : str
        Identifier (flavor name or ablation label) substituted into the recipe metadata.
    accelerator : str
        Lightning accelerator to normalize the recipe's train block to.
    """
    # initialize from this foundation (repo-relative; anvil runs from root)
    params = recipe["procedure"]["model"]["params"]
    params["from_foundation"] = foundation_rel

    # freeze the MPNN so finetuning measures representation quality, not initialization luck;
    # the FFN head (ffn_lr) still trains freely
    params["mpnn_lr"] = 0

    # normalize the accelerator: baselines pin a laptop device, cluster nodes resolve "auto"
    train_params = recipe.get("procedure", {}).get("train", {}).get("params")
    if isinstance(train_params, dict) and "accelerator" in train_params:
        train_params["accelerator"] = accelerator

    # relabel machine identifiers from the backbone to the label
    meta = recipe.get("metadata", {})
    for key in ("name", "tag"):
        if isinstance(meta.get(key), str):
            meta[key] = meta[key].replace(BACKBONE, label)
    if isinstance(meta.get("tags"), list):
        meta["tags"] = [label if tag == BACKBONE else tag for tag in meta["tags"]]
    return recipe


def _endpoint_name(template: Path) -> str:
    """Strip the backbone prefix so files sit under ``configs/<flavor>/<endpoint>.yaml``."""
    stem = template.stem
    prefix = f"{BACKBONE}_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def _generate_one(templates: list[Path], out_dir: Path, foundation_rel: str,
                  label: str, accelerator: str) -> int:
    """Write one retargeted recipe per template into ``out_dir``; return the count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for template in templates:
        recipe = yaml.safe_load(template.read_text())
        recipe = _retarget(recipe, foundation_rel, label, accelerator)
        (out_dir / f"{_endpoint_name(template)}.yaml").write_text(
            yaml.safe_dump(recipe, sort_keys=False)
        )
    return len(templates)


def main() -> None:
    """Generate per-flavor recipes, or per-ablation recipes for one explicit foundation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavors", nargs="*", default=None, help="flavor subset (default all)")
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR, help="templates dir")
    parser.add_argument("--accelerator", default="auto", help="lightning accelerator")
    parser.add_argument(
        "--foundation",
        type=Path,
        default=None,
        help="ablation mode: explicit foundation checkpoint (requires --out-subdir)",
    )
    parser.add_argument(
        "--out-subdir",
        default=None,
        help="ablation mode: output dir name under configs/ and the recipe label",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    templates = sorted(args.baseline_dir.glob(f"{BACKBONE}_*.yaml"))
    if not templates:
        raise SystemExit(f"no baseline templates in {args.baseline_dir}")

    # ablation mode: one explicit foundation -> configs/<out-subdir>/
    if args.foundation is not None:
        if not args.out_subdir:
            raise SystemExit("--foundation requires --out-subdir")
        foundation_rel = str(args.foundation.resolve().relative_to(REPO_ROOT))
        n = _generate_one(
            templates, CONFIGS_DIR / args.out_subdir, foundation_rel, args.out_subdir,
            args.accelerator,
        )
        logger.info("ablation %s: %d recipes -> %s", args.out_subdir, n, args.out_subdir)
        return

    # flavor mode: one recipe set per registry flavor
    flavors = args.flavors or flavor_names()
    n_written = 0
    for flavor in flavors:
        foundation_rel = str(foundation_path(flavor).relative_to(REPO_ROOT))
        n = _generate_one(templates, CONFIGS_DIR / flavor, foundation_rel, flavor, args.accelerator)
        n_written += n
        logger.info("flavor %s: %d recipes", flavor, n)
    logger.info("wrote %d recipes across %d flavors", n_written, len(flavors))


if __name__ == "__main__":
    main()
