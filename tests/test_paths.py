"""Tests for the on-disk path helpers (flavor corpus and ablation namespacing)."""

from sarizard.analysis import paths


def test_surrogate_flavor_uses_its_own_corpus():
    assert paths.flavor_corpus("surrogate_adme") == paths.SURROGATE_CORPUS_SMILES


def test_other_flavors_use_the_shared_corpus():
    assert paths.flavor_corpus("osmordred") == paths.CORPUS_SMILES


def test_ablation_label_is_namespaced():
    assert paths.ablation_label("full") == "ablation_full"


def test_ablation_foundation_name_matches_label():
    assert paths.ablation_foundation_name("order_fix") == "ablation_order_fix_mp.pt"


def test_ablation_prescaled_zarr_under_ablations_cache():
    path = paths.ablation_prescaled_zarr("full")

    assert path == paths.ABLATIONS_CACHE_DIR / "full" / "prescaled.zarr"


def test_foundation_path_for_flavor():
    assert paths.foundation_path("ecfp") == paths.FOUNDATIONS_DIR / "ecfp_mp.pt"
