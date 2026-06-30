"""Tests for the shared chunk-based train/val split helper."""

import numpy as np
import pytest

from sarizard.pretraining.splitting import chunk_row_ranges, train_val_chunk_indices


def test_last_chunk_excluded_from_both_sides():
    train, val = train_val_chunk_indices(100)

    assigned = set(train.tolist()) | set(val.tolist())
    assert 99 not in assigned
    assert assigned == set(range(99))


def test_train_and_val_are_disjoint():
    train, val = train_val_chunk_indices(100)

    assert set(train.tolist()).isdisjoint(val.tolist())


def test_both_sides_sorted_ascending():
    train, val = train_val_chunk_indices(100)

    assert list(train) == sorted(train)
    assert list(val) == sorted(val)


@pytest.mark.parametrize(
    ("n_chunks", "expected_train"),
    [(100, 90), (1000, 900), (40, 36)],
    ids=["100-chunks", "1000-chunks", "40-chunks"],
)
def test_train_size_is_floor_of_fraction_times_count(n_chunks: int, expected_train: int):
    train, _ = train_val_chunk_indices(n_chunks, train_frac=0.9)

    assert len(train) == expected_train


def test_split_is_deterministic_across_calls():
    first_train, first_val = train_val_chunk_indices(500)
    second_train, second_val = train_val_chunk_indices(500)

    assert np.array_equal(first_train, second_train)
    assert np.array_equal(first_val, second_val)


def test_different_seed_changes_partition():
    _, val_a = train_val_chunk_indices(500, seed=42)
    _, val_b = train_val_chunk_indices(500, seed=7)

    assert not np.array_equal(val_a, val_b)


def test_chunk_row_ranges_expands_to_half_open_blocks():
    ranges = chunk_row_ranges(np.array([0, 2, 5]), rows_per_chunk=128)

    assert ranges == [(0, 128), (256, 384), (640, 768)]
