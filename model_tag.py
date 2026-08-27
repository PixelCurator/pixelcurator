"""Single source of truth for the CLIP model config and the embedding-space tag.

Every stage that loads CLIP (01_embed, 03_classify_cluster, 06_taxonomy,
07_incremental, embed_test, tests/test_ml_smoke) imports MODEL_NAME /
PRETRAINED from here, so the whole pipeline switches model config in exactly
one place and can never mix configs across stages (#38).

The sidecar tag file (`<embeddings>.model.json`, written next to the `.npy`)
records which model config produced the stored embedding space. Writers
(01_embed resume-append, 07_incremental append) and readers that compare
freshly-encoded CLIP output against the stored space (03_classify_cluster,
06_taxonomy) must call `check_tag()` first and hard-fail on a mismatch:
appending or comparing embeddings from a different model config silently
corrupts every cosine similarity downstream. A missing sidecar next to
existing artefacts is treated as a mismatch too -- legacy untagged artefacts
must be re-embedded (or manually tagged), never silently adopted.
"""
import json
from pathlib import Path

# OpenAI CLIP weights were trained with the QuickGELU activation; the plain
# "ViT-B-32" config paired with pretrained="openai" is a documented
# accuracy-degrading mismatch (open_clip README, mlfoundations/open_clip;
# open_clip warns about it at every load). Measured on this library (#38):
# top-10 neighbour overlap 0.745 between the two spaces, 35% different
# top-1 matches, 84.9% zero-shot label agreement -- the divergence is real.
MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"


class EmbeddingSpaceMismatchError(RuntimeError):
    """Stored embeddings were produced by a different CLIP model config."""


def tag_path(emb_path: Path) -> Path:
    """Sidecar path for an embeddings file: clip_vitb32.npy -> clip_vitb32.model.json"""
    return emb_path.with_suffix(".model.json")


def read_tag(emb_path: Path) -> dict | None:
    p = tag_path(emb_path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def write_tag(emb_path: Path) -> None:
    tag_path(emb_path).write_text(
        json.dumps({"model": MODEL_NAME, "pretrained": PRETRAINED})
    )


def check_tag(emb_path: Path) -> None:
    """Hard-fail unless the stored embedding space matches the current config.

    Cold start (no embeddings file) passes -- there is no space to mismatch.
    """
    if not Path(emb_path).exists():
        return
    tag = read_tag(emb_path)
    current = {"model": MODEL_NAME, "pretrained": PRETRAINED}
    if tag is None:
        raise EmbeddingSpaceMismatchError(
            f"{emb_path} exists but has no model tag ({tag_path(emb_path)} missing). "
            f"Refusing to touch an untagged embedding space: it may have been "
            f"produced by a different model config than the current "
            f"{current}. Either re-embed from scratch (delete the .npy and its "
            f"uuid index, then run 01_embed.py) or, if you are certain the "
            f"artefacts match the current config, write the tag manually: "
            f"python -c \"import model_tag, pathlib; "
            f"model_tag.write_tag(pathlib.Path('{emb_path}'))\""
        )
    if tag != current:
        raise EmbeddingSpaceMismatchError(
            f"Embedding space mismatch for {emb_path}: stored artefacts were "
            f"produced with {tag}, but the current code uses {current}. "
            f"Mixing configs silently corrupts every cosine similarity. "
            f"Re-embed from scratch: delete {emb_path}, its uuid index and "
            f"{tag_path(emb_path)}, then run 01_embed.py."
        )
