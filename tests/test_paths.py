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


def test_ablation_variant_label_carries_the_seed():
    assert paths.ablation_variant_label("order_fix", 7) == "ablation_order_fix__s7"


def test_ablation_variant_foundation_name_carries_the_seed():
    assert paths.ablation_variant_foundation_name("full", 42) == "ablation_full__s42_mp.pt"


def test_parse_ablation_variant_splits_name_and_seed():
    assert paths.parse_ablation_variant("ablation_order_fix__s7") == ("order_fix", 7)


def test_parse_ablation_variant_without_seed_returns_none():
    # a plain ablation label (no __s<seed>) round-trips to its name with no seed
    assert paths.parse_ablation_variant("ablation_full") == ("full", None)


def test_seed_variant_label_and_parse_round_trip():
    label = paths.seed_variant_label("ecfp", 3)

    assert label == "ecfp__s3"
    assert paths.parse_seed_variant(label) == ("ecfp", 3)


def test_parse_seed_variant_without_seed_returns_none():
    assert paths.parse_seed_variant("ecfp") == ("ecfp", None)


def test_flavor_variant_label_and_foundation_path():
    assert paths.flavor_variant_label("ecfp", 42) == "ecfp__s42"
    assert paths.foundation_variant_path("ecfp", 42) == paths.FOUNDATIONS_DIR / "ecfp__s42_mp.pt"


def test_parse_seed_variant_collapses_prefixed_bases():
    # an LR-experiment label keeps its prefixed base intact, just stripping the seed
    assert paths.parse_seed_variant("lr_reduced__ecfp__s2") == ("lr_reduced__ecfp", 2)


def test_ablation_prescaled_zarr_under_ablations_cache():
    path = paths.ablation_prescaled_zarr("full")

    assert path == paths.ABLATIONS_CACHE_DIR / "full" / "prescaled.zarr"


def test_foundation_path_for_flavor():
    assert paths.foundation_path("ecfp") == paths.FOUNDATIONS_DIR / "ecfp_mp.pt"
