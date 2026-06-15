from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision import datasets

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inference import XRayInference  # noqa: E402
from train_xray_cnn import (  # noqa: E402
    ChestPneumoniaCNN,
    _build_eval_transform,
    preview_gradcam_samples,
    resolve_data_root,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate balanced Grad-CAM previews from saved checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=SCRIPT_DIR / "outputs" / "xray" / "xray_cnn.pt")
    parser.add_argument("--data-root", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = list(payload["class_names"])
    image_size = int(payload["image_size"])
    model = ChestPneumoniaCNN(num_classes=int(payload["num_classes"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    data_root = resolve_data_root(args.data_root)
    test_ds = datasets.ImageFolder(str(data_root / "test"), transform=_build_eval_transform(image_size))
    output_dir = args.checkpoint.parent
    preview_gradcam_samples(model, test_ds, class_names, device, output_dir, image_size, args.samples)
    print(f"Done. Previews in {output_dir / 'test_previews'}")


if __name__ == "__main__":
    main()
