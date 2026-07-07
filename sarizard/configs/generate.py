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
    python -m sarizard.configs.generate                      # all flavors, all endpoints
    python -m sarizard.configs.generate --flavors ecfp jazzy # a subset of flavors
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from sarizard.analysis.paths import (
    CONFIGS_DIR,
    REPO_ROOT,
    foundation_variant_path,
    seed_variant_label,
)
from sarizard.pretraining.flavors import flavor_names

logger = logging.getLogger(__name__)

BACKBONE = "chemeleon"  # token the baseline recipes use in names, tags, and from_foundation
BASELINE_DIR = CONFIGS_DIR / "_baseline"

# finetune protocols for the MPNN backbone, set as a multiple of the recipe's ffn_lr:
# frozen (the default sweep) holds the backbone fixed; reduced and unlocked are the LR
# experiments that let it adapt (see TODO.md and run_lr_experiments.sh)
MPNN_LR_MODES = ("frozen", "reduced", "unlocked")
REDUCED_MPNN_LR_FACTOR = 0.1


def _mpnn_lr(ffn_lr: float, mode: str) -> float:
    """Return the MPNN learning rate for a finetune protocol, relative to ``ffn_lr``."""
    if mode == "frozen":
        return 0
    if mode == "reduced":
        return ffn_lr * REDUCED_MPNN_LR_FACTOR
    if mode == "unlocked":
        return ffn_lr
    raise ValueError(f"unknown mpnn_lr_mode {mode!r}; choose from {MPNN_LR_MODES}")


def _retarget(
    recipe: dict,
    foundation_rel: str,
    label: str,
    accelerator: str,
    *,
    mpnn_lr_mode: str = "frozen",
) -> dict:
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
    mpnn_lr_mode : {"frozen", "reduced", "unlocked"}, optional
        The backbone finetune protocol. ``frozen`` (default) holds the MPNN fixed
        (``mpnn_lr=0``); ``reduced`` and ``unlocked`` set it to a fraction of, or equal to,
        the recipe's ``ffn_lr`` for the learning-rate experiments.
    """
    # initialize from this foundation (repo-relative; anvil runs from root)
    params = recipe["procedure"]["model"]["params"]
    params["from_foundation"] = foundation_rel

    # set the backbone learning rate by protocol; frozen (default) measures representation
    # quality without initialization luck, the others let the MPNN coadapt with the FFN head
    params["mpnn_lr"] = _mpnn_lr(params.get("ffn_lr", 0.0), mpnn_lr_mode)

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
                  label: str, accelerator: str, *, mpnn_lr_mode: str = "frozen") -> int:
    """Write one retargeted recipe per template into ``out_dir``; return the count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for template in templates:
        recipe = yaml.safe_load(template.read_text())
        recipe = _retarget(recipe, foundation_rel, label, accelerator, mpnn_lr_mode=mpnn_lr_mode)
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
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="flavor mode: training seeds; one recipe set per (flavor, seed) variant",
    )
    parser.add_argument(
        "--mpnn-lr-mode", default="frozen", choices=MPNN_LR_MODES,
        help="backbone finetune protocol (frozen, or reduced/unlocked for an LR experiment); "
        "applies to both flavor and ablation modes",
    )
    parser.add_argument(
        "--label-prefix", default="",
        help="flavor mode: namespace prefix for the recipe label/dir (e.g. lr_reduced)",
    )
    parser.add_argument(
        "--stock-baseline", action="store_true",
        help="write configs/chemeleon_stock/<endpoint>.yaml: the baseline templates "
        "unchanged (from_foundation stays 'chemeleon', so anvil downloads and finetunes "
        "the released stock checkpoint) except relabeled for provenance. This is the "
        "external reference column for the report card, not one of our pretrained "
        "flavors; run once, it does not depend on the corpus or pretraining regime.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    templates = sorted(args.baseline_dir.glob(f"{BACKBONE}_*.yaml"))
    if not templates:
        raise SystemExit(f"no baseline templates in {args.baseline_dir}")

    # stock-baseline mode: relabel only, from_foundation stays "chemeleon" (a no-op override)
    if args.stock_baseline:
        label = "chemeleon_stock"
        n = _generate_one(
            templates, CONFIGS_DIR / label, BACKBONE, label, args.accelerator,
            mpnn_lr_mode=args.mpnn_lr_mode,
        )
        logger.info("stock baseline (%s): %d recipes -> configs/%s", args.mpnn_lr_mode, n, label)
        return

    # ablation mode: one explicit foundation -> configs/<out-subdir>/
    if args.foundation is not None:
        if not args.out_subdir:
            raise SystemExit("--foundation requires --out-subdir")
        foundation_rel = str(args.foundation.resolve().relative_to(REPO_ROOT))
        n = _generate_one(
            templates, CONFIGS_DIR / args.out_subdir, foundation_rel, args.out_subdir,
            args.accelerator, mpnn_lr_mode=args.mpnn_lr_mode,
        )
        logger.info(
            "ablation %s (%s): %d recipes -> %s",
            args.out_subdir,
            args.mpnn_lr_mode,
            n,
            args.out_subdir,
        )
        return

    # flavor mode: one recipe set per (flavor, seed). With --label-prefix and --mpnn-lr-mode
    # this also drives the LR experiments off the same flavor foundations (lr_<mode>__<flavor>).
    flavors = args.flavors or flavor_names()
    n_written = 0
    for flavor in flavors:
        base = f"{args.label_prefix}__{flavor}" if args.label_prefix else flavor
        for seed in args.seeds:
            label = seed_variant_label(base, seed)
            foundation_rel = str(foundation_variant_path(flavor, seed).relative_to(REPO_ROOT))
            n = _generate_one(
                templates, CONFIGS_DIR / label, foundation_rel, label, args.accelerator,
                mpnn_lr_mode=args.mpnn_lr_mode,
            )
            n_written += n
            logger.info("flavor %s seed %d (%s): %d recipes", flavor, seed, args.mpnn_lr_mode, n)
    logger.info(
        "wrote %d recipes across %d flavors x %d seeds", n_written, len(flavors), len(args.seeds)
    )


if __name__ == "__main__":
    main()
