from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inference import XRayInference  # noqa: E402
from train_xray_cnn import resolve_data_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test X-ray pneumonia model on official test split.")
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=SCRIPT_DIR)
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    test_root = data_root / "test"
    xray = XRayInference.from_default_paths(SCRIPT_DIR)

    correct = 0
    total = 0
    for class_name in xray._class_names:
        class_dir = test_root / class_name
        if not class_dir.is_dir():
            print(f"Missing class folder: {class_dir}")
            continue
        images = sorted(class_dir.glob("*.*"))[: args.samples_per_class]
        for image_path in images:
            out = xray.predict_image_path(image_path)
            ok = out["predicted_class"] == class_name
            correct += int(ok)
            total += 1
            status = "OK" if ok else "MISS"
            print(
                f"[{status}] {image_path.name} | true={class_name} "
                f"pred={out['predicted_class']} conf={out['confidence']:.3f}"
            )

    if total == 0:
        raise SystemExit("No test images found.")
    acc = correct / total
    print(f"\nSmoke test accuracy: {correct}/{total} = {acc:.1%}")
    if acc < 0.8:
        raise SystemExit("Smoke test accuracy below 80%.")


if __name__ == "__main__":
    main()
