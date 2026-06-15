"""Export raw skin lesion PNGs from HMNIST CSV (not Grad-CAM previews)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SKIN_OUT = Path(__file__).resolve().parent / "skin"
CSV = ROOT / "Heart&SkinCancer" / "Skin Cancer Data set" / "hmnist_28_28_RGB.csv"
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"]
EXPORT_SIZE = 128
PER_CLASS = 2


def _load_hmnist(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    pixels = df.iloc[:, :-1].to_numpy(dtype=np.uint8)
    labels = df.iloc[:, -1].to_numpy(dtype=np.int64)
    side = int(round((pixels.shape[1] // 3) ** 0.5))
    images = pixels.reshape(-1, side, side, 3)
    return images, labels


def main() -> None:
    if not CSV.is_file():
        raise FileNotFoundError(f"HMNIST CSV not found: {CSV}. Download to Heart&SkinCancer/Skin Cancer Data set/.")

    images, labels = _load_hmnist(CSV)
    SKIN_OUT.mkdir(parents=True, exist_ok=True)
    for old in SKIN_OUT.glob("*"):
        if old.is_file():
            old.unlink()

    manifest_skin: list[dict[str, str]] = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        indices = np.flatnonzero(labels == class_idx)
        if indices.size == 0:
            continue
        pick = indices[:PER_CLASS]
        for n, idx in enumerate(pick, start=1):
            suffix = "" if n == 1 else f"_{n:02d}"
            filename = f"lesion_{class_name}{suffix}.jpg"
            out_path = SKIN_OUT / filename
            img = Image.fromarray(images[idx], mode="RGB").resize(
                (EXPORT_SIZE, EXPORT_SIZE),
                Image.Resampling.BILINEAR,
            )
            img.save(out_path, quality=92)
            manifest_skin.append(
                {
                    "file": f"skin/{filename}",
                    "expected_class": class_name,
                    "note": f"Raw HMNIST lesion ({class_name})",
                }
            )
            print(f"Wrote {out_path.name} (label={class_name})")

    manifest_path = Path(__file__).resolve().parent / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["skin"] = manifest_skin
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {manifest_path.name} skin entries ({len(manifest_skin)} files).")


if __name__ == "__main__":
    main()
