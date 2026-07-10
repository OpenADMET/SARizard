"""Shared metric and ordering constants for evaluation and the report card.

Kept dependency-free (no openadmet, no plotting) so both the heavy ``evaluate`` step and the
light ``report_card`` step import the same definitions without dragging the training stack.
"""

from __future__ import annotations

# ordered (recipe-prefix -> dataset) rules, most-specific prefix first so "asap_potency" matches
# before "asap" and the single-task cyp1a2 recipe (cyp1a2_st) matches before the multi-task cyp
# recipe (cyp_mt). Both CYP recipes map to the one openadmet_cyp group: cyp1a2 is scored both by
# a dedicated single-task model and as one head of the multi-task model, and both belong to the
# same openadmet CYP source
_DATASET_RULES = (
    ("asap_potency", "asap_potency"),
    ("asap", "asap"),
    ("biogen", "biogen"),
    ("chembl", "chembl"),
    ("cyp1a2", "openadmet_cyp"),
    ("cyp", "openadmet_cyp"),
    ("expansionrx", "expansionrx"),
    ("herg", "herg"),
    ("pxr", "pxr"),
)

# dataset groups in report-card row order (the rule labels, de-duplicated, first appearance kept)
DATASETS = tuple(dict.fromkeys(label for _, label in _DATASET_RULES))

# metric columns produced by evaluate.py, in output order
METRIC_COLUMNS = ("r2", "rmse", "mae", "mse", "spearman", "kendall", "rae")

# metrics where a larger value is better (drives report-card color direction)
HIGHER_IS_BETTER = frozenset({"r2", "spearman", "kendall"})

# display labels for axes and titles
METRIC_LABELS = {
    "r2": "R²",
    "rmse": "RMSE",
    "mae": "MAE",
    "mse": "MSE",
    "spearman": "Spearman ρ",
    "kendall": "Kendall τ",
    "rae": "RAE",
}


def dataset_of(recipe: str) -> str:
    """Return the dataset group a recipe name belongs to (most-specific prefix rule wins)."""
    for prefix, label in _DATASET_RULES:
        if recipe == prefix or recipe.startswith(f"{prefix}_"):
            return label
    return recipe.split("_")[0]
