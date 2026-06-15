"""Run quick smoke tests on bundled sample inputs (no full datasets required)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = Path(__file__).resolve().parent
MANIFEST = SAMPLES / "manifest.json"


def _load_inference(module_dir: Path, module_name: str):
    path = module_dir / "inference.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print("=== Multimodal smoke tests ===\n")

    xray_mod = _load_inference(ROOT / "XRay-Pneumonia", "xray_inference_smoke")
    heart_skin_mod = _load_inference(ROOT / "Heart&SkinCancer", "heart_skin_inference_smoke")

    xray = xray_mod.XRayInference.from_default_paths(ROOT / "XRay-Pneumonia")
    xray_ok = xray_miss = 0
    for item in manifest["xray"]:
        path = SAMPLES / item["file"]
        expected = item["expected"]
        out = xray.predict_image_path(path)
        ok = out["predicted_class"] == expected
        xray_ok += int(ok)
        xray_miss += int(not ok)
        mark = "OK" if ok else "MISS"
        print(f"[{mark}] X-ray {path.name}: pred={out['predicted_class']} conf={out['confidence']:.3f} (expected {expected})")

    heart = heart_skin_mod.HeartInference.from_default_paths(ROOT / "Heart&SkinCancer")
    for item in manifest["heart"]:
        path = SAMPLES / item["file"]
        features = json.loads(path.read_text(encoding="utf-8"))
        out = heart.predict_features(features)
        print(f"[OK] Heart {path.name}: pred={out['predicted_class']} conf={out['confidence']:.3f}")

    skin = heart_skin_mod.SkinInference.from_default_paths(ROOT / "Heart&SkinCancer")
    skin_ok = skin_miss = 0
    for item in manifest["skin"]:
        path = SAMPLES / item["file"]
        out = skin.predict_bytes(path.read_bytes())
        expected = item.get("expected_class")
        if expected:
            ok = out["predicted_class"] == expected
            skin_ok += int(ok)
            skin_miss += int(not ok)
            mark = "OK" if ok else "MISS"
            print(
                f"[{mark}] Skin {path.name}: pred={out['predicted_class']} conf={out['confidence']:.3f} (expected {expected})"
            )
        else:
            print(f"[OK] Skin {path.name}: pred={out['predicted_class']} conf={out['confidence']:.3f}")

    print(f"\nX-ray summary: {xray_ok}/{xray_ok + xray_miss} matched expected labels.")
    if skin_ok + skin_miss:
        print(f"Skin summary: {skin_ok}/{skin_ok + skin_miss} matched expected labels.")
    print("Done.")


if __name__ == "__main__":
    main()
