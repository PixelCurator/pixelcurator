"""ML pipeline smoke tests: prove the exact pinned wheels in requirements.txt
(torch, torchvision, numpy, Pillow, onnxruntime, open_clip_torch, nudenet)
actually work together, not just that `pip install` succeeded.

These run only in .github/workflows/ml-smoke.yml, which installs the full
requirements.txt. The everyday tests.yml job installs just pytest + numpy
(see its header comment) and never touches torch/torchvision/onnxruntime --
each heavy import below goes through `pytest.importorskip`, so this module
is cleanly SKIPPED (not errored) when those packages aren't installed. That
keeps the fast, lazy-import-only acceptance suite in tests.yml green and
untouched.

See issue #28: a Dependabot bump to torch/torchvision/numpy/onnxruntime/
Pillow could previously merge on green CI even though nothing exercised the
real embedding/NSFW-detection pipeline (ABI mismatch, a resolver picking a
torch/torchvision pair that doesn't match the hard pin, a numpy major bump
breaking a compiled extension, ...). Each test below targets one specific
way that failure mode shows up:

  * test_clip_zero_shot_classification_is_correct -- catches a pipeline
    that imports fine and runs cleanly but returns semantically wrong
    output (the "didn't crash, but garbage" failure mode a bare import
    check would miss entirely).
  * test_torch_numpy_roundtrip_identity -- catches a numpy ABI break, the
    boundary a numpy major-version bump can silently corrupt.
  * test_torchvision_nms_matches_expected_boxes -- crosses the compiled
    torch/torchvision C++ ABI boundary that requirements.txt's hard pin
    (torchvision==0.28.0 requires torch==2.13.0 exactly) exists to protect.
  * test_nudedetector_via_02_nsfw_production_path -- runs 02_nsfw.py itself
    (unmodified, as a subprocess against a synthetic sandbox) so
    NudeDetector().detect() is exercised through onnxruntime via the exact
    file production code takes, not a hand-rolled substitute.
"""
import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")
open_clip = pytest.importorskip("open_clip")
np = pytest.importorskip("numpy")
PIL_Image = pytest.importorskip("PIL.Image")
pytest.importorskip("onnxruntime")

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── 1. CLIP zero-shot classification: catches "runs but returns garbage" ──────

def test_clip_zero_shot_classification_is_correct():
    """Load CLIP ViT-B-32 -- the exact production call in 01_embed.py /
    03_classify_cluster.py -- and classify solid-color synthetic images
    against text prompts. A pipeline that imports fine and produces *some*
    output but a broken/garbage embedding space would still pass a bare
    "did it crash" check; this doesn't."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    red = PIL_Image.new("RGB", (224, 224), (220, 20, 20))
    blue = PIL_Image.new("RGB", (224, 224), (20, 20, 220))
    images = torch.stack([preprocess(red), preprocess(blue)])
    prompts = ["a photo of something red", "a photo of something blue"]

    with torch.no_grad():
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = model.encode_text(tokenizer(prompts))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Same LOGIT_SCALE convention as 03_classify_cluster.py
        similarity = (image_features @ text_features.T) * 100.0
        probs = similarity.softmax(dim=-1)

    top1 = probs.argmax(dim=-1).tolist()
    assert top1 == [1, 0], (  # DELIBERATELY WRONG -- proving the gate catches this; reverted next commit
        f"expected red image -> 'red' prompt (idx 0) and blue image -> "
        f"'blue' prompt (idx 1), got top-1 indices {top1} "
        f"(probs={probs.tolist()})"
    )
    # Not just "the right one wins" but wins by a wide, confident margin --
    # a badly-wired but not totally broken pipeline can still eke out a
    # narrow argmax.
    assert probs[0, 0].item() > 0.9
    assert probs[1, 1].item() > 0.9


# ── 2. torch<->numpy roundtrip: catches a numpy ABI break ─────────────────────

def test_torch_numpy_roundtrip_identity():
    """torch.Tensor.numpy() and torch.from_numpy() cross the numpy C ABI --
    exactly what a numpy major-version bump (requirements.txt pins
    numpy>=2.5.1) can silently break for a compiled extension built against
    a different ABI."""
    original = [[1.5, -2.25, 3.0], [4.0, 5.5, -6.75]]
    tensor = torch.tensor(original, dtype=torch.float64)

    as_numpy = tensor.numpy()
    assert isinstance(as_numpy, np.ndarray)
    assert np.array_equal(as_numpy, np.array(original, dtype=np.float64))

    back_to_torch = torch.from_numpy(as_numpy)
    assert torch.equal(tensor, back_to_torch)
    assert back_to_torch.dtype == torch.float64


# ── 3. torchvision.ops.nms: crosses the compiled torch/torchvision ABI ────────

def test_torchvision_nms_matches_expected_boxes():
    """torchvision's compiled ops (nms is a C++ extension registered against
    a specific torch build) are exactly what requirements.txt's hard pin
    (torchvision==0.28.0 requires torch==2.13.0 exactly) protects against. A
    mismatched pair either throws on import/op-lookup or silently returns
    wrong indices -- both are failure modes this asserts against."""
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],    # box A
            [1.0, 1.0, 11.0, 11.0],    # box B -- heavily overlaps A, lower score
            [50.0, 50.0, 60.0, 60.0],  # box C -- disjoint, highest score
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.8, 0.95])

    keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
    kept = set(keep.tolist())

    # C (disjoint, highest score) and A (higher-scored of the overlapping
    # pair) survive; B is suppressed by A.
    assert kept == {0, 2}, f"expected boxes {{0, 2}} kept, got {kept}"


# ── 4. NudeDetector via the real production import path in 02_nsfw.py ─────────

def test_nudedetector_via_02_nsfw_production_path(tmp_path):
    """Runs 02_nsfw.py itself, unmodified, as a subprocess against a
    synthetic one-photo sandbox -- so `from nudenet import NudeDetector`,
    `NudeDetector()`, and `.detect()` (all module-level code in 02_nsfw.py,
    executed unconditionally -- there is no `if __name__ == "__main__":`
    guard) run through onnxruntime exactly as they do in the real pipeline,
    not through a hand-rolled substitute call in this test file.

    02_nsfw.py hardcodes `ROOT = Path.home() / "photo-sort"` (unlike
    04_contact_sheets/06_retrain/07_incremental/08_import_photos_albums, it
    doesn't read $PIXEL_ROOT) -- so the sandbox is built under an overridden
    $HOME for this subprocess only, the same test-isolation approach
    conftest.py already uses to keep tests off the real photo-sort
    directory.
    """
    fake_home = tmp_path / "home"
    root = fake_home / "photo-sort"
    metadata = root / "metadata"
    metadata.mkdir(parents=True)

    image_path = tmp_path / "synthetic.jpg"
    PIL_Image.new("RGB", (320, 320), (128, 180, 90)).save(image_path, "JPEG")

    uuid = "TESTUUID-0000-0000-0000-000000000001"
    with (metadata / "inventory.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "uuid", "original_filename", "date", "ismissing", "has_local_original",
            "has_local_derivative", "derivative_path", "derivative_size_bytes",
            "uti_original", "isphoto", "ismovie",
        ])
        writer.writerow([
            uuid, "synthetic.jpg", "2024-01-01", "False", "False", "True",
            str(image_path), str(image_path.stat().st_size),
            "public.jpeg", "True", "False",
        ])

    env = os.environ.copy()
    env["HOME"] = str(fake_home)

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "02_nsfw.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"02_nsfw.py exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    out_csv = metadata / "nsfw_scores.csv"
    assert out_csv.exists(), "02_nsfw.py did not write metadata/nsfw_scores.csv"

    with out_csv.open() as f:
        rows = {row["uuid"]: row for row in csv.DictReader(f)}

    assert uuid in rows, f"no row for {uuid} in nsfw_scores.csv: {rows}"
    row = rows[uuid]
    assert row["nsfw_label"] in {"safe", "nude", "explicit"}, (
        f"unexpected nsfw_label {row['nsfw_label']!r} -- 02_nsfw.py writes "
        f"'error' when NudeDetector().detect() raised inside its except "
        f"clause (see stderr above for the real exception)"
    )
    # Sanity: confidence must parse as a real number (a Python exception
    # string surviving into this column would fail this).
    float(row["nsfw_confidence"])
