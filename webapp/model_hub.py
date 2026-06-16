from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HEART_SKIN_DIR = ROOT / "Heart&SkinCancer"
XRAY_DIR = ROOT / "XRay-Pneumonia"


def _load_xray_inference_class():
    module_path = XRAY_DIR / "inference.py"
    if not module_path.is_file():
        raise ImportError(f"X-ray inference module not found: {module_path}")
    spec = importlib.util.spec_from_file_location("xray_pneumonia_inference", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load X-ray inference module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.XRayInference


class ModelHub:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._xray = None
        self._heart = None
        self._skin = None
        self._load_xray()
        self._load_heart_skin()

    def availability(self) -> dict[str, bool]:
        return {
            "xray": self._xray is not None,
            "heart": self._heart is not None,
            "skin": self._skin is not None,
        }

    def _load_xray(self) -> None:
        try:
            xray_cls = _load_xray_inference_class()
            self._xray = xray_cls.from_default_paths(XRAY_DIR)
        except Exception:
            self._xray = None

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

    def predict_xray_bytes(self, data: bytes) -> dict[str, Any]:
        if self._xray is None:
            return {
                "available": False,
                "error": "Chest X-ray model checkpoint not found. Run XRay-Pneumonia/train_xray_cnn.py first.",
            }

        out = self._xray.predict_bytes(data)
        predicted_class = out["predicted_class"]
        confidence = float(out["confidence"])
        ranked = sorted(out["probabilities"].items(), key=lambda item: item[1], reverse=True)
        predicted_labels = [predicted_class] if predicted_class == "PNEUMONIA" else []

        return {
            "available": True,
            "display_text": f"{predicted_class} ({confidence:.2f})",
            "predicted_labels": predicted_labels,
            "predicted_class": predicted_class,
            "headline_confidence": confidence,
            "top5": [{"label": label, "confidence": float(prob)} for label, prob in ranked],
            "probabilities": out["probabilities"],
            "gradcam_image": out.get("gradcam_image"),
        }

    def predict_skin_bytes(self, data: bytes) -> dict[str, Any]:
        if self._skin is None:
            return {"available": False, "error": "Skin model checkpoint not found."}

        out = self._skin.predict_bytes(data)
        ranked = sorted(out["probabilities"].items(), key=lambda item: item[1], reverse=True)
        predicted_class = out["predicted_class"]
        confidence = float(out["confidence"])
        return {
            "available": True,
            "display_text": f"{predicted_class} ({confidence:.2f})",
            "predicted_class": predicted_class,
            "confidence": confidence,
            "headline_confidence": confidence,
            "top3": [{"label": c, "confidence": float(p)} for c, p in out["top3"]],
            "top5": [{"label": label, "confidence": float(prob)} for label, prob in ranked[:5]],
            "probabilities": out["probabilities"],
            # Grad-CAM removed for skin outputs to avoid rendering on the frontend
        }

    def predict_heart(self, features: dict[str, Any]) -> dict[str, Any]:
        if self._heart is None:
            return {"available": False, "error": "Heart model checkpoint/preprocessor not found."}

        out = self._heart.predict_features(features)
        probs = out.get("probabilities", {})
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        predicted_class = out["predicted_class"]
        confidence = float(out["confidence"])
        return {
            "available": True,
            "display_text": f"{predicted_class} ({confidence:.2f})",
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probability": confidence,
            "headline_confidence": confidence,
            "top2": [{"label": c, "confidence": float(p)} for c, p in ranked[:2]],
            "probabilities": probs,
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
