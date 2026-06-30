"""Shared metric and ordering constants for evaluation and the report card.

Kept dependency-free (no openadmet, no plotting) so both the heavy ``evaluate`` step and the
light ``report_card`` step import the same definitions without dragging the training stack.
"""

from __future__ import annotations

# dataset groups in report-card row order; longest-prefix first so "asap_potency" matches
# before "asap" and "cyp1a2" before "cyp" when parsing a recipe name
DATASETS = (
    "asap_potency", "asap", "biogen", "chembl", "cyp1a2", "cyp", "expansionrx", "herg", "pxr",
)

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
    """Return the dataset group a recipe name belongs to (longest matching prefix)."""
    for dataset in DATASETS:
        if recipe == dataset or recipe.startswith(f"{dataset}_"):
            return dataset
    return recipe.split("_")[0]
