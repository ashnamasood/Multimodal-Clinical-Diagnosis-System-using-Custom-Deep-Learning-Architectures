from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inference import XRayInference  # noqa: E402

xray = XRayInference.from_default_paths(SCRIPT_DIR)
test_root = SCRIPT_DIR / "chest_xray" / "chest_xray" / "test"
input_root = SCRIPT_DIR.parent / "input"

print("=== Official test: PNEUMONIA ===")
for path in sorted((test_root / "PNEUMONIA").glob("*.jpeg"))[:5]:
    out = xray.predict_image_path(path)
    print(f"{path.name}: {out['predicted_class']} ({out['confidence']:.3f})")

print("\n=== Official test: NORMAL ===")
for path in sorted((test_root / "NORMAL").glob("*.jpeg"))[:3]:
    out = xray.predict_image_path(path)
    print(f"{path.name}: {out['predicted_class']} ({out['confidence']:.3f})")

print("\n=== Legacy input/ folder ===")
if input_root.is_dir():
    for path in sorted(input_root.glob("*.*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        out = xray.predict_image_path(path)
        print(f"{path.name}: {out['predicted_class']} ({out['confidence']:.3f}) | {out['probabilities']}")
