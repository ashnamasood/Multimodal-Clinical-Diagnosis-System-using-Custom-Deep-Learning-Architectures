from __future__ import annotations

"""
Simulate dashboard "user upload" flows (no train/test split): raw form-like dict + image file/bytes.

Run from repo root:
  python "Heart&SkinCancer/simulate_user_uploads.py"

Optional:
  python "Heart&SkinCancer/simulate_user_uploads.py" --skin-image "path/to/photo.png"
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd

from inference import HeartInference, SkinInference

DEFAULT_HEART_CSV = SCRIPT_DIR / "Heart Data set" / "heart.csv"


def _sample_heart_row(csv_path: Path) -> dict:
    series = pd.read_csv(csv_path).iloc[0]
    return {c: series[c] for c in series.index if c != "HeartDisease"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate user uploads for heart + skin models.")
    p.add_argument(
        "--heart-csv",
        type=Path,
        default=DEFAULT_HEART_CSV,
        help="Use first data row (features only) as fake form submission.",
    )
    p.add_argument(
        "--skin-image",
        type=Path,
        default=None,
        help="Lesion image path. Default: first PNG under outputs/skin/val_previews if present.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    skin_path = args.skin_image
    if skin_path is None:
        previews = sorted((SCRIPT_DIR / "outputs" / "skin" / "val_previews").glob("*.png"))
        skin_path = previews[0] if previews else None

    print("=== Simulated user upload / inference ===\n")

    heart = HeartInference.from_default_paths()
    row = _sample_heart_row(args.heart_csv)
    h_out = heart.predict_features(row)
    print("Heart (as if user submitted form fields from first CSV row, no label):")
    print(json.dumps({k: h_out[k] for k in ("predicted_class", "confidence", "probabilities")}, indent=2))

    print()
    skin = SkinInference.from_default_paths()
    if skin_path is None or not skin_path.is_file():
        print("Skin: no image path (pass --skin-image or train with --gradcam-samples to create val_previews).")
        return

    with open(skin_path, "rb") as fh:
        data = fh.read()
    s_out = skin.predict_bytes(data)
    print(f"Skin (as if user uploaded file: {skin_path.name}):")
    print(json.dumps({k: s_out[k] for k in ("predicted_class", "confidence", "top3")}, indent=2))


if __name__ == "__main__":
    main()
