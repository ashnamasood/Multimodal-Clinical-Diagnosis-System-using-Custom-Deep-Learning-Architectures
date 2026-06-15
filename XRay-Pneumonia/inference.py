from __future__ import annotations

"""
Production-style inference for chest X-ray pneumonia classification.

Usage::

    xray = XRayInference.from_default_paths()
    result = xray.predict_bytes(request.FILES['chest_xray'].read())
"""

import base64
import io
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_xray_cnn import (  # noqa: E402
    ChestPneumoniaCNN,
    GradCAM,
    _build_eval_transform,
    _get_last_conv_layer,
    _make_gradcam_overlay,
)


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class XRayInference:
    def __init__(self, checkpoint_path: Path, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"X-ray checkpoint not found: {checkpoint_path}")

        payload = _torch_load(checkpoint_path, self.device)
        self._class_names: list[str] = list(payload["class_names"])
        self._image_size = int(payload["image_size"])
        num_classes = int(payload["num_classes"])
        self._model = ChestPneumoniaCNN(num_classes=num_classes).to(self.device)
        self._model.load_state_dict(payload["model_state_dict"])
        self._model.eval()
        self._transform = _build_eval_transform(self._image_size)
        self._gradcam = GradCAM(self._model, _get_last_conv_layer(self._model))

    def _gradcam_data_url(self, rgb: Image.Image, tensor: torch.Tensor, class_index: int) -> str | None:
        try:
            display = rgb.resize((self._image_size, self._image_size), Image.Resampling.BILINEAR)
            cam_input = tensor.clone().detach().requires_grad_(True)
            heatmap = self._gradcam(cam_input, class_index)
            overlay = _make_gradcam_overlay(display, heatmap)
            buf = io.BytesIO()
            overlay.save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/png;base64,{encoded}"
        except Exception:
            return None

    @classmethod
    def from_default_paths(cls, base: Path | None = None) -> XRayInference:
        root = base or SCRIPT_DIR
        return cls(root / "outputs" / "xray" / "xray_cnn.pt")

    def predict_pil(self, image: Image.Image) -> dict[str, Any]:
        rgb = image.convert("RGB")
        tensor = self._transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_i = int(logits.argmax(dim=1).item())
            ranked = sorted(
                [(self._class_names[i], float(probs[i].item())) for i in range(len(self._class_names))],
                key=lambda x: x[1],
                reverse=True,
            )
        result = {
            "predicted_class": self._class_names[pred_i],
            "predicted_index": pred_i,
            "confidence": float(probs[pred_i].item()),
            "probabilities": {name: prob for name, prob in ranked},
            "top3": ranked[:3],
        }
        gradcam_image = self._gradcam_data_url(rgb, tensor, pred_i)
        if gradcam_image:
            result["gradcam_image"] = gradcam_image
        return result

    def predict_image_path(self, path: Path | str) -> dict[str, Any]:
        with Image.open(path) as img:
            return self.predict_pil(img)

    def predict_bytes(self, data: bytes) -> dict[str, Any]:
        return self.predict_pil(Image.open(io.BytesIO(data)))
