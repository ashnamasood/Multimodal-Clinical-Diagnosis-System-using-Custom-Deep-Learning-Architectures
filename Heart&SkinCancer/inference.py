from __future__ import annotations

"""
Production-style inference for the dashboard: load checkpoints once, run on user-supplied data.

Heart: dict or one-row frame with columns Age, Sex, ChestPainType, RestingBP, Cholesterol,
FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope (no HeartDisease).

Skin: file path, raw bytes (e.g. request.FILES['lesion'].read()), or PIL Image (RGB).

Django-style sketch::

    heart = HeartInference.from_default_paths()
    skin = SkinInference.from_default_paths()
    h = heart.predict_features({...})
    s = skin.predict_bytes(request.FILES['skin'].read())
"""

import base64
import io
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_heart_tabular import (  # noqa: E402
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    HeartMLP,
)
from train_skin_cnn import (  # noqa: E402
    GradCAM,
    SkinLesionCNN,
    _build_eval_transform,
    _get_last_conv_layer,
    _make_gradcam_overlay,
)

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class HeartInference:
    def __init__(
        self,
        checkpoint_path: Path,
        preprocessor_path: Path,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Heart checkpoint not found: {checkpoint_path}")
        if not preprocessor_path.is_file():
            raise FileNotFoundError(f"Heart preprocessor not found: {preprocessor_path}")

        payload = _torch_load(checkpoint_path, self.device)
        self._class_names: list[str] = list(payload["class_names"])
        self._model = HeartMLP(
            input_dim=int(payload["input_dim"]),
            hidden1=int(payload["hidden1"]),
            hidden2=int(payload["hidden2"]),
            embed_dim=int(payload["embed_dim"]),
            num_classes=int(payload["num_classes"]),
        ).to(self.device)
        self._model.load_state_dict(payload["model_state_dict"])
        self._model.eval()
        self._pre = joblib.load(preprocessor_path)

    @classmethod
    def from_default_paths(cls, base: Path | None = None) -> HeartInference:
        root = base or SCRIPT_DIR
        return cls(
            root / "outputs" / "heart" / "heart_mlp.pt",
            root / "outputs" / "heart" / "heart_preprocessor.joblib",
        )

    @torch.no_grad()
    def predict_features(self, row: dict[str, Any]) -> dict[str, Any]:
        """One patient: all training feature keys (omit HeartDisease)."""
        missing = [c for c in FEATURE_COLS if c not in row]
        if missing:
            raise ValueError(f"Missing heart features: {missing}")

        frame = pd.DataFrame([{k: row[k] for k in FEATURE_COLS}])
        x = self._pre.transform(frame)
        tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        logits = self._model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_i = int(logits.argmax(dim=1).item())
        emb = self._model.forward_embedding(tensor).squeeze(0).cpu().tolist()

        return {
            "predicted_class": self._class_names[pred_i],
            "predicted_index": pred_i,
            "confidence": float(probs[pred_i].item()),
            "probabilities": {self._class_names[i]: float(probs[i].item()) for i in range(len(self._class_names))},
            "embedding_dim32": emb,
        }


class SkinInference:
    def __init__(self, checkpoint_path: Path, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Skin checkpoint not found: {checkpoint_path}")

        payload = _torch_load(checkpoint_path, self.device)
        self._class_names: list[str] = list(payload["class_names"])
        self._image_size = int(payload["image_size"])
        num_classes = int(payload["num_classes"])
        self._model = SkinLesionCNN(num_classes=num_classes).to(self.device)
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
    def from_default_paths(cls, base: Path | None = None) -> SkinInference:
        root = base or SCRIPT_DIR
        return cls(root / "outputs" / "skin" / "skin_cnn.pt")

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
            "probabilities": {name: p for name, p in ranked},
            "top3": ranked[:3],
        }
        gradcam_image = self._gradcam_data_url(rgb, tensor, pred_i)
        if gradcam_image:
            result["gradcam_image"] = gradcam_image
        return result

    def predict_image_path(self, path: Path) -> dict[str, Any]:
        with Image.open(path) as img:
            return self.predict_pil(img)

    def predict_bytes(self, data: bytes) -> dict[str, Any]:
        return self.predict_pil(Image.open(io.BytesIO(data)))
