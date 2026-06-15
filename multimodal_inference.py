"""
Unified Multimodal Clinical Diagnosis System
Integrates chest X-ray, skin cancer, and heart disease models.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
XRAY_DIR = ROOT / "XRay-Pneumonia"


def _load_xray_inference():
    module_path = XRAY_DIR / "inference.py"
    spec = importlib.util.spec_from_file_location("xray_pneumonia_inference", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.XRayInference


class MultimodalDiagnosisSystem:
    """Unified interface for diagnostic models."""

    def __init__(
        self,
        chest_xray_checkpoint: Path | str | None = None,
        device: str = "cpu",
    ) -> None:
        del device
        self._xray = None
        checkpoint = Path(chest_xray_checkpoint) if chest_xray_checkpoint else XRAY_DIR / "outputs" / "xray" / "xray_cnn.pt"
        try:
            xray_cls = _load_xray_inference()
            if checkpoint.exists():
                self._xray = xray_cls(checkpoint)
                print(f"Chest X-ray model loaded from {checkpoint}")
            else:
                print(f"Chest X-ray checkpoint not found: {checkpoint}")
        except Exception as exc:
            print(f"Chest X-ray model not loaded: {exc}")

    def diagnose_chest_xray(self, image_path: Path | str) -> dict[str, Any]:
        if self._xray is None:
            return {"error": "Chest X-ray model not loaded"}

        image_path = Path(image_path)
        if not image_path.exists():
            return {"error": f"Image not found: {image_path}"}

        out = self._xray.predict_image_path(image_path)
        predicted_class = out["predicted_class"]
        findings = []
        if predicted_class == "PNEUMONIA":
            findings.append({"disease": "PNEUMONIA", "confidence": out["confidence"]})

        return {
            "modality": "Chest X-ray",
            "image_path": str(image_path),
            "predicted_class": predicted_class,
            "confidence": out["confidence"],
            "findings": findings,
            "all_confidences": out["probabilities"],
            "num_findings": len(findings),
            "summary": (
                f"Detected pneumonia (confidence {out['confidence']:.2f})"
                if findings
                else f"No pneumonia detected ({predicted_class}, confidence {out['confidence']:.2f})"
            ),
        }

    def multimodal_diagnosis(self, chest_xray_path: Path | str | None = None) -> dict[str, Any]:
        results: dict[str, Any] = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "chest_xray": None,
        }
        if chest_xray_path:
            results["chest_xray"] = self.diagnose_chest_xray(chest_xray_path)
        results["summary"] = self._generate_summary(results)
        return results

    def _generate_summary(self, results: dict[str, Any]) -> str:
        if not results["chest_xray"] or "error" in results["chest_xray"]:
            return "Unable to process chest X-ray."

        if results["chest_xray"].get("num_findings", 0) > 0:
            return "Concerning chest X-ray finding: pneumonia detected. Recommend clinical follow-up."
        return "No pneumonia detected on chest X-ray."


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Clinical Diagnosis")
    parser.add_argument("--chest-xray", type=str, help="Path to chest X-ray image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(XRAY_DIR / "outputs" / "xray" / "xray_cnn.pt"),
        help="Path to chest X-ray checkpoint",
    )
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    system = MultimodalDiagnosisSystem(chest_xray_checkpoint=args.checkpoint, device=args.device)
    results = system.multimodal_diagnosis(chest_xray_path=args.chest_xray)

    print("\n" + "=" * 70)
    print("MULTIMODAL CLINICAL DIAGNOSIS REPORT")
    print("=" * 70)
    print(json.dumps(results, indent=2))
    print("=" * 70)
    print(f"\n{results['summary']}\n")


if __name__ == "__main__":
    main()
