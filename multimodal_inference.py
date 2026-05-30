"""
Unified Multimodal Clinical Diagnosis System
Integrates: Chest X-ray, Skin Cancer, and Heart Disease Models
"""

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
import json
from torchvision import transforms, models

# Import chest X-ray model directly
from train_chest_xray import ChestXRayCNN


class MultimodalDiagnosisSystem:
    """Unified interface for all three diagnostic models."""

    CHEST_XRAY_CLASSES = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
    ]

    def __init__(
        self,
        chest_xray_checkpoint: Path | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.chest_xray_model = None
        self.chest_xray_transform = None
        self.chest_xray_thresholds = {}

        # Load Chest X-ray model
        if chest_xray_checkpoint and Path(chest_xray_checkpoint).exists():
            self._load_chest_xray(chest_xray_checkpoint)
        else:
            print("⚠️  Chest X-ray model not loaded (checkpoint not found)")

    def _load_chest_xray(self, checkpoint_path: Path | str):
        """Load chest X-ray model and metadata."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load class names from JSON if available
        classes_json = checkpoint_path.parent / "chest_xray_classes.json"
        if classes_json.exists():
            with open(classes_json) as f:
                self.chest_xray_classes = json.load(f)
        else:
            self.chest_xray_classes = self.CHEST_XRAY_CLASSES

        # Build and load model
        num_classes = len(self.chest_xray_classes)
        # Load class names from checkpoint first, then JSON
        if "class_names" in checkpoint:
            self.chest_xray_classes = checkpoint["class_names"]
        else:
            classes_json = checkpoint_path.parent / "chest_xray_classes.json"
            if classes_json.exists():
                with open(classes_json) as f:
                    self.chest_xray_classes = json.load(f)
            else:
                self.chest_xray_classes = self.CHEST_XRAY_CLASSES

        # Build and load model with correct num_classes
        num_classes = len(self.chest_xray_classes)
        self.chest_xray_model = ChestXRayCNN(num_classes=num_classes, pretrained=False)
        self.chest_xray_model.load_state_dict(checkpoint["model_state_dict"])

        # Load thresholds from checkpoint
        if "optimized_thresholds" in checkpoint:
            self.chest_xray_thresholds = checkpoint["optimized_thresholds"]
        self.chest_xray_model.to(self.device)
        self.chest_xray_model.eval()

        # Load thresholds if available (from test_metrics.json)
        metrics_json = checkpoint_path.parent / "test_metrics.json"
        if metrics_json.exists():
            with open(metrics_json) as f:
                metrics = json.load(f)
                per_label = metrics.get("per_label", {})
                for class_name in self.chest_xray_classes:
                    if class_name in per_label:
                        self.chest_xray_thresholds[class_name] = per_label[class_name].get("threshold", 0.5)
                    else:
                        self.chest_xray_thresholds[class_name] = 0.5
        else:
            # Load from test_metrics.json as fallback
            metrics_json = checkpoint_path.parent / "test_metrics.json"
            if metrics_json.exists():
                with open(metrics_json) as f:
                    metrics = json.load(f)
                    per_label = metrics.get("per_label", {})
                    for class_name in self.chest_xray_classes:
                        if class_name in per_label:
                            self.chest_xray_thresholds[class_name] = per_label[class_name].get("threshold", 0.5)
                        else:
                            self.chest_xray_thresholds[class_name] = 0.5
            else:
                # Default thresholds
                for class_name in self.chest_xray_classes:
                    self.chest_xray_thresholds[class_name] = 0.5

        # Setup image transformation
        self.chest_xray_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"✅ Chest X-ray model loaded ({num_classes} classes, {len(self.chest_xray_thresholds)} thresholds)")

    def diagnose_chest_xray(self, image_path: Path | str) -> dict[str, Any]:
        """Diagnose from chest X-ray image."""
        if not self.chest_xray_model:
            return {"error": "Chest X-ray model not loaded"}

        image_path = Path(image_path)
        if not image_path.exists():
            return {"error": f"Image not found: {image_path}"}

        # Load and preprocess image
        try:
            image = Image.open(image_path).convert("RGB")
            x = self.chest_xray_transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            return {"error": f"Failed to load image: {e}"}

        # Inference
        with torch.no_grad():
            logits = self.chest_xray_model(x)
            probs = torch.sigmoid(logits).squeeze(0)

        # Apply thresholds
        findings = []
        confidences = {}
        for i, class_name in enumerate(self.chest_xray_classes):
            prob = float(probs[i].item())
            confidences[class_name] = prob
            threshold = float(self.chest_xray_thresholds.get(class_name, 0.5))
            if prob >= threshold:
                findings.append((class_name, prob))

        findings.sort(key=lambda x: x[1], reverse=True)

        return {
            "modality": "Chest X-ray",
            "image_path": str(image_path),
            "findings": [{"disease": name, "confidence": conf} for name, conf in findings],
            "all_confidences": confidences,
            "num_findings": len(findings),
            "summary": f"Detected {len(findings)} finding(s)" if findings else "No significant findings",
        }

    def multimodal_diagnosis(
        self,
        chest_xray_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Run diagnosis on chest X-ray."""
        results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "chest_xray": None,
        }

        # Chest X-ray
        if chest_xray_path:
            results["chest_xray"] = self.diagnose_chest_xray(chest_xray_path)

        # Summary
        results["summary"] = self._generate_summary(results)

        return results

    def _generate_summary(self, results: dict[str, Any]) -> str:
        """Generate clinical summary."""
        if not results["chest_xray"] or "error" in results["chest_xray"]:
            return "⚠️  Unable to process chest X-ray."

        num_findings = results["chest_xray"].get("num_findings", 0)
        if num_findings >= 3:
            return "🔴 Multiple findings detected. Recommend immediate clinical consultation."
        elif num_findings > 0:
            return "🟡 Concerning findings detected. Recommend clinical follow-up."
        else:
            return "🟢 No significant pathological findings."


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Multimodal Clinical Diagnosis")
    parser.add_argument("--chest-xray", type=str, help="Path to chest X-ray image")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    # Initialize system
    system = MultimodalDiagnosisSystem(
        chest_xray_checkpoint="outputs/chest_xray_cnn.pt",
        device=args.device,
    )

    # Run diagnosis
    results = system.multimodal_diagnosis(
        chest_xray_path=args.chest_xray,
    )

    # Pretty print results
    print("\n" + "="*70)
    print("MULTIMODAL CLINICAL DIAGNOSIS REPORT")
    print("="*70)
    print(json.dumps(results, indent=2))
    print("="*70)
    print(f"\n{results['summary']}\n")


if __name__ == "__main__":
    main()
