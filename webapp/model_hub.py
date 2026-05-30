from __future__ import annotations

import importlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
HEART_SKIN_DIR = ROOT / "Heart&SkinCancer"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_chest_xray import (  # noqa: E402
    ChestXRayCNN,
    _format_clinical_prediction,
    _load_checkpoint_payload,
    _load_thresholds_for_checkpoint,
    _resolve_checkpoint_path,
)


class ModelHub:
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        self._xray_model: ChestXRayCNN | None = None
        self._xray_class_names: list[str] = []
        self._xray_thresholds: dict[str, float] = {}
        self._xray_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self._heart = None
        self._skin = None

        self._load_xray()
        self._load_heart_skin()

    def availability(self) -> dict[str, bool]:
        return {
            "xray": self._xray_model is not None,
            "heart": self._heart is not None,
            "skin": self._skin is not None,
        }

    def _load_xray(self) -> None:
        checkpoint_path = ROOT / "outputs" / "chest_xray_cnn.pt"
        resolved = _resolve_checkpoint_path(checkpoint_path)
        if not resolved.exists():
            return

        payload = _load_checkpoint_payload(resolved, self.device)
        class_names = payload.get("class_names")
        if not isinstance(class_names, list) or not class_names:
            return

        model = ChestXRayCNN(num_classes=len(class_names), pretrained=False).to(self.device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()

        self._xray_model = model
        self._xray_class_names = class_names
        self._xray_thresholds = _load_thresholds_for_checkpoint(resolved, class_names)

    def _load_heart_skin(self) -> None:
        if str(HEART_SKIN_DIR) not in sys.path:
            sys.path.insert(0, str(HEART_SKIN_DIR))
        try:
            inf = importlib.import_module("inference")
        except Exception:
            return

        try:
            self._heart = inf.HeartInference.from_default_paths(HEART_SKIN_DIR)
        except Exception:
            self._heart = None

        try:
            self._skin = inf.SkinInference.from_default_paths(HEART_SKIN_DIR)
        except Exception:
            self._skin = None

    @torch.no_grad()
    def predict_xray_bytes(self, data: bytes) -> dict[str, Any]:
        if self._xray_model is None:
            return {"available": False, "error": "Chest X-ray model checkpoint not found."}

        image = Image.open(io.BytesIO(data)).convert("RGB")
        tensor = self._xray_transform(image).unsqueeze(0).to(self.device)
        logits = self._xray_model(tensor)
        probs_t = torch.sigmoid(logits).squeeze(0).cpu().tolist()

        display_text, labels, _, headline_conf = _format_clinical_prediction(
            probs_t,
            self._xray_class_names,
            self._xray_thresholds,
        )

        ranked = sorted(zip(self._xray_class_names, probs_t), key=lambda x: x[1], reverse=True)
        return {
            "available": True,
            "display_text": display_text,
            "predicted_labels": labels,
            "headline_confidence": float(headline_conf),
            "top5": [{"label": k, "confidence": float(v)} for k, v in ranked[:5]],
            "thresholds": self._xray_thresholds,
        }

    def predict_skin_bytes(self, data: bytes) -> dict[str, Any]:
        if self._skin is None:
            return {"available": False, "error": "Skin model checkpoint not found."}

        out = self._skin.predict_bytes(data)
        return {
            "available": True,
            "predicted_class": out["predicted_class"],
            "confidence": float(out["confidence"]),
            "top3": [{"label": c, "confidence": float(p)} for c, p in out["top3"]],
        }

    def predict_heart(self, features: dict[str, Any]) -> dict[str, Any]:
        if self._heart is None:
            return {"available": False, "error": "Heart model checkpoint/preprocessor not found."}

        out = self._heart.predict_features(features)
        probs = out.get("probabilities", {})
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return {
            "available": True,
            "predicted_class": out["predicted_class"],
            "confidence": float(out["confidence"]),
            "top2": [{"label": c, "confidence": float(p)} for c, p in ranked[:2]],
        }

    def fused_assessment(self, xray: dict[str, Any] | None, skin: dict[str, Any] | None, heart: dict[str, Any] | None) -> dict[str, Any]:
        scores: list[float] = []
        notes: list[str] = []

        if xray and xray.get("available"):
            x_score = float(xray.get("headline_confidence", 0.0))
            if xray.get("predicted_labels"):
                scores.append(x_score)
                notes.append(f"X-ray positive findings: {', '.join(xray['predicted_labels'][:3])}")

        if skin and skin.get("available"):
            cls = str(skin.get("predicted_class", "")).lower()
            conf = float(skin.get("confidence", 0.0))
            malignant_keywords = ("mel", "bcc", "scc", "cancer", "malig")
            if any(k in cls for k in malignant_keywords):
                scores.append(conf)
                notes.append(f"Skin high-risk class: {skin.get('predicted_class')}")
            else:
                scores.append(max(0.0, 1 - conf) * 0.3)

        if heart and heart.get("available"):
            cls = str(heart.get("predicted_class", "")).lower()
            conf = float(heart.get("confidence", 0.0))
            if "heart disease" in cls and "no" not in cls:
                scores.append(conf)
                notes.append("Heart model indicates disease risk.")
            else:
                scores.append(max(0.0, 1 - conf) * 0.3)

        if not scores:
            return {
                "overall_risk_score": 0.0,
                "overall_risk_level": "unknown",
                "recommendation": "Upload at least one valid modality.",
                "notes": notes,
            }

        score = float(sum(scores) / len(scores))
        if score >= 0.7:
            level = "high"
            rec = "Immediate clinical follow-up is recommended."
        elif score >= 0.4:
            level = "moderate"
            rec = "Clinical review is recommended."
        else:
            level = "low"
            rec = "No urgent risk from current inputs; continue monitoring."

        return {
            "overall_risk_score": score,
            "overall_risk_level": level,
            "recommendation": rec,
            "notes": notes,
        }


HEART_FORM_SCHEMA = {
    "Sex": ["M", "F"],
    "ChestPainType": ["TA", "ATA", "NAP", "ASY"],
    "RestingECG": ["Normal", "ST", "LVH"],
    "ExerciseAngina": ["Y", "N"],
    "ST_Slope": ["Up", "Flat", "Down"],
    "numeric": ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"],
}


def parse_heart_json(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("heart_features must be a JSON object.")
    return payload
