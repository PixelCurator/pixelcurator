"""Unit tests: 06_retrain.py business logic.

Tests for build_training_set, MIN_CORR threshold, and weak-label filtering.
Pure Python — no server, no subprocess.
"""
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

# Allow importing 06_retrain from the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ── helpers ───────────────────────────────────────────────────────────────────

def make_embeddings(n: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, dim)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e


SKIP_ALBUMS = {"Thrash", "_test_"}
MIN_CORR = 3


def build_training_set(corrections, assigned, emb, uuid_to_idx,
                       weak_threshold=0.80):
    """Replicate 06_retrain.build_training_set() for isolated unit testing."""
    corr_counts = Counter(corrections.values())
    valid_corr_albums = {a for a, n in corr_counts.items() if n >= MIN_CORR}

    X, y, weights = [], [], []

    for uid, album in corrections.items():
        if album not in valid_corr_albums:
            continue
        if uid not in uuid_to_idx:
            continue
        X.append(emb[uuid_to_idx[uid]])
        y.append(album)
        weights.append(2.0)

    for uid, (album, conf, src) in assigned.items():
        if uid in corrections:
            continue
        if conf < weak_threshold:
            continue
        if album.endswith("-unsure") or album in SKIP_ALBUMS:
            continue
        if uid not in uuid_to_idx:
            continue
        X.append(emb[uuid_to_idx[uid]])
        y.append(album)
        weights.append(1.0)

    return np.array(X, dtype=np.float32), np.array(y), np.array(weights)


# ── MIN_CORR threshold ────────────────────────────────────────────────────────

class TestMinCorrThreshold:
    def test_album_below_min_corr_excluded(self):
        """Albums with fewer than MIN_CORR corrections must not become labels."""
        uuids = [f"UUID-{i:03d}" for i in range(10)]
        emb = make_embeddings(10)
        uuid_to_idx = {u: i for i, u in enumerate(uuids)}

        # "Garten" has 2 corrections (< MIN_CORR=3), "Kochen" has 3
        corrections = {
            uuids[0]: "Garten",
            uuids[1]: "Garten",   # only 2 → excluded
            uuids[2]: "Kochen",
            uuids[3]: "Kochen",
            uuids[4]: "Kochen",   # exactly 3 → included
        }
        assigned = {}
        X, y, w = build_training_set(corrections, assigned, emb, uuid_to_idx)
        assert "Garten" not in set(y)
        assert "Kochen" in set(y)
        assert len(X) == 3  # only Kochen's 3 samples

    def test_album_exactly_at_min_corr_included(self):
        uuids = [f"UUID-{i:03d}" for i in range(5)]
        emb = make_embeddings(5)
        uuid_to_idx = {u: i for i, u in enumerate(uuids)}
        corrections = {uuids[i]: "Garten" for i in range(MIN_CORR)}
        X, y, _ = build_training_set(corrections, {}, emb, uuid_to_idx)
        assert set(y) == {"Garten"}
        assert len(X) == MIN_CORR


class TestStrongWeakLabels:
    def test_corrections_get_weight_2(self):
        """Strong labels (corrections) must have weight=2.0."""
        uuids = [f"UUID-{i:03d}" for i in range(3)]
        emb = make_embeddings(3)
        uuid_to_idx = {u: i for i, u in enumerate(uuids)}
        corrections = {u: "Garten" for u in uuids}
        _, _, weights = build_training_set(corrections, {}, emb, uuid_to_idx)
        assert all(w == 2.0 for w in weights)

    def test_assigned_get_weight_1(self):
        """Weak labels (original assignments) must have weight=1.0."""
        uuids = [f"UUID-{i:03d}" for i in range(4)]
        emb = make_embeddings(4)
        uuid_to_idx = {u: i for i, u in enumerate(uuids)}
        corrections = {}
        assigned = {u: ("Kochen", 0.95, "retrain") for u in uuids}
        _, _, weights = build_training_set(corrections, assigned, emb, uuid_to_idx)
        assert all(w == 1.0 for w in weights)

    def test_corrections_take_precedence_over_assigned(self):
        """A UUID with a correction must not appear twice in the training set."""
        uuids = [f"UUID-{i:03d}" for i in range(5)]
        emb = make_embeddings(5)
        uuid_to_idx = {u: i for i, u in enumerate(uuids)}
        # All 5 are in both corrections (3 needed for valid album) and assigned
        corrections = {uuids[i]: "Garten" for i in range(3)}
        # Some of those also appear in assigned with a different label
        assigned = {uuids[i]: ("Kochen", 0.95, "retrain") for i in range(3)}
        _, y, _ = build_training_set(corrections, assigned, emb, uuid_to_idx)
        # Only "Garten" should appear (corrections win); "Kochen" excluded
        assert "Kochen" not in set(y)
        assert "Garten" in set(y)


class TestWeakLabelFiltering:
    def test_low_confidence_assigned_excluded(self):
        """Assigned photos below WEAK_THRESHOLD must be excluded from training."""
        uuids = [f"UUID-{i:03d}" for i in range(4)]
        emb = make_embeddings(4)
        uuid_to_idx = {u: i for i, u in enumerate(uuids)}
        corrections = {}
        # conf 0.79 → below WEAK_THRESHOLD 0.80 → excluded
        assigned = {
            uuids[0]: ("Garten", 0.50, "retrain"),   # too low
            uuids[1]: ("Garten", 0.79, "retrain"),   # just below threshold
            uuids[2]: ("Garten", 0.80, "retrain"),   # at threshold → included
            uuids[3]: ("Garten", 0.95, "retrain"),   # well above → included
        }
        X, y, _ = build_training_set(corrections, assigned, emb, uuid_to_idx,
                                     weak_threshold=0.80)
        assert len(X) == 2  # only uuids[2] and uuids[3]

    def test_unsure_albums_excluded_from_weak_labels(self):
        """Albums ending in -unsure must not be used as weak labels."""
        uuids = [f"UUID-{i:03d}" for i in range(3)]
        emb = make_embeddings(3)
        uuid_to_idx = {u: i for i, u in enumerate(uuids)}
        assigned = {u: ("Garten-unsure", 0.95, "retrain") for u in uuids}
        X, y, _ = build_training_set({}, assigned, emb, uuid_to_idx)
        assert len(X) == 0  # all excluded

    def test_thrash_excluded_from_weak_labels(self):
        uuids = [f"UUID-{i:03d}" for i in range(3)]
        emb = make_embeddings(3)
        uuid_to_idx = {u: i for i, u in enumerate(uuids)}
        assigned = {u: ("Thrash", 0.99, "retrain") for u in uuids}
        X, y, _ = build_training_set({}, assigned, emb, uuid_to_idx)
        assert len(X) == 0
