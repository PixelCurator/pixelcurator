"""Unit tests for model_tag.py -- the embedding-space tag guard (#38).

Runs in the fast lane (tests.yml): model_tag has no heavy imports, so the
guard semantics are pinned on every PR, not just in the ml-smoke workflow.
The end-to-end proof that 01_embed.py refuses to resume-append onto a
mismatched space lives in tests/test_ml_smoke.py (needs torch).
"""
import json
from pathlib import Path

import pytest

import model_tag


def _fake_space(tmp_path: Path) -> Path:
    emb = tmp_path / "clip_vitb32.npy"
    emb.write_bytes(b"\x93NUMPY fake")
    return emb


class TestTagPath:
    def test_sidecar_sits_next_to_npy(self, tmp_path):
        emb = tmp_path / "clip_vitb32.npy"
        assert model_tag.tag_path(emb) == tmp_path / "clip_vitb32.model.json"


class TestWriteRead:
    def test_write_then_read_roundtrip(self, tmp_path):
        emb = _fake_space(tmp_path)
        model_tag.write_tag(emb)
        assert model_tag.read_tag(emb) == {
            "model": model_tag.MODEL_NAME,
            "pretrained": model_tag.PRETRAINED,
        }

    def test_read_missing_sidecar_is_none(self, tmp_path):
        assert model_tag.read_tag(tmp_path / "clip_vitb32.npy") is None


class TestCheckTag:
    def test_cold_start_passes(self, tmp_path):
        """No embeddings file -> nothing to mismatch -> no error."""
        model_tag.check_tag(tmp_path / "clip_vitb32.npy")

    def test_matching_tag_passes(self, tmp_path):
        emb = _fake_space(tmp_path)
        model_tag.write_tag(emb)
        model_tag.check_tag(emb)

    def test_untagged_legacy_artefacts_hard_fail(self, tmp_path):
        """Embeddings without a sidecar must NOT be silently adopted: they
        may predate the tag mechanism and stem from a different model
        config. This is the resume-append trap from #38."""
        emb = _fake_space(tmp_path)
        with pytest.raises(model_tag.EmbeddingSpaceMismatchError, match="no model tag"):
            model_tag.check_tag(emb)

    def test_different_model_hard_fails(self, tmp_path):
        emb = _fake_space(tmp_path)
        model_tag.tag_path(emb).write_text(
            json.dumps({"model": "ViT-B-32-some-other-config",
                        "pretrained": model_tag.PRETRAINED})
        )
        with pytest.raises(model_tag.EmbeddingSpaceMismatchError, match="mismatch"):
            model_tag.check_tag(emb)

    def test_different_pretrained_hard_fails(self, tmp_path):
        emb = _fake_space(tmp_path)
        model_tag.tag_path(emb).write_text(
            json.dumps({"model": model_tag.MODEL_NAME, "pretrained": "laion2b"})
        )
        with pytest.raises(model_tag.EmbeddingSpaceMismatchError, match="mismatch"):
            model_tag.check_tag(emb)


class TestModelConfigPin:
    def test_config_is_quickgelu_variant(self):
        """OpenAI CLIP weights were trained with QuickGELU; loading them via
        the plain "ViT-B-32" config degrades the whole embedding space
        (measured: top-10 neighbour overlap 0.745, 35% different top-1
        matches, see #38). An accidental revert to the plain config must
        fail CI here, in the fast lane, on every PR."""
        assert model_tag.MODEL_NAME == "ViT-B-32-quickgelu"
        assert model_tag.PRETRAINED == "openai"
