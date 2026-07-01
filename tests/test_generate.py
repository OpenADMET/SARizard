"""Tests for per-flavor and per-ablation finetuning recipe generation."""

from pathlib import Path

import yaml

from sarizard.configs.generate import _endpoint_name, _generate_one, _retarget


def _baseline_recipe() -> dict:
    """A minimal recipe with the fields _retarget rewrites."""
    return {
        "procedure": {
            "model": {
                "params": {"from_foundation": "chemeleon", "mpnn_lr": 1.0e-3, "ffn_lr": 1.0e-3}
            },
            "train": {"params": {"accelerator": "mps"}},
        },
        "metadata": {
            "name": "chemeleon_cyp",
            "tag": "chemeleon-cyp",
            "tags": ["chemeleon", "cyp"],
        },
    }


def test_retarget_points_at_foundation_and_freezes_mpnn():
    out = _retarget(_baseline_recipe(), "foundations/ecfp_mp.pt", "ecfp", "auto")

    params = out["procedure"]["model"]["params"]
    assert params["from_foundation"] == "foundations/ecfp_mp.pt"
    assert params["mpnn_lr"] == 0


def test_retarget_reduced_mode_sets_fraction_of_ffn_lr():
    out = _retarget(
        _baseline_recipe(), "foundations/ecfp__s1_mp.pt", "ecfp__s1", "auto",
        mpnn_lr_mode="reduced",
    )

    # reduced backbone LR is one tenth of the recipe's ffn_lr (1e-3 -> 1e-4)
    assert out["procedure"]["model"]["params"]["mpnn_lr"] == 1.0e-4


def test_retarget_unlocked_mode_matches_ffn_lr():
    out = _retarget(
        _baseline_recipe(), "foundations/ecfp__s1_mp.pt", "ecfp__s1", "auto",
        mpnn_lr_mode="unlocked",
    )

    assert out["procedure"]["model"]["params"]["mpnn_lr"] == 1.0e-3


def test_retarget_normalizes_accelerator():
    out = _retarget(_baseline_recipe(), "foundations/ecfp_mp.pt", "ecfp", "auto")

    assert out["procedure"]["train"]["params"]["accelerator"] == "auto"


def test_retarget_relabels_backbone_to_label():
    out = _retarget(_baseline_recipe(), "foundations/ecfp_mp.pt", "ecfp", "auto")

    meta = out["metadata"]
    assert meta["name"] == "ecfp_cyp"
    assert meta["tag"] == "ecfp-cyp"
    assert meta["tags"] == ["ecfp", "cyp"]


def test_retarget_relabels_with_ablation_label():
    out = _retarget(_baseline_recipe(), "foundations/ablation_full_mp.pt", "ablation_full", "auto")

    assert out["metadata"]["name"] == "ablation_full_cyp"


def test_endpoint_name_strips_backbone_prefix():
    assert _endpoint_name(Path("chemeleon_cyp_mt.yaml")) == "cyp_mt"


def test_endpoint_name_passes_through_unprefixed_name():
    assert _endpoint_name(Path("custom_recipe.yaml")) == "custom_recipe"


def test_generate_one_writes_one_recipe_per_template(tmp_path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    for stem in ("chemeleon_a", "chemeleon_b"):
        (template_dir / f"{stem}.yaml").write_text(yaml.safe_dump(_baseline_recipe()))
    templates = sorted(template_dir.glob("*.yaml"))
    out_dir = tmp_path / "out"

    count = _generate_one(templates, out_dir, "foundations/ablation_full_mp.pt",
                          "ablation_full", "auto")

    assert count == 2
    written = sorted(p.name for p in out_dir.glob("*.yaml"))
    assert written == ["a.yaml", "b.yaml"]
    recipe = yaml.safe_load((out_dir / "a.yaml").read_text())
    params = recipe["procedure"]["model"]["params"]
    assert params["from_foundation"] == "foundations/ablation_full_mp.pt"
    assert params["mpnn_lr"] == 0
