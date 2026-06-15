from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inference import XRayInference  # noqa: E402
from train_xray_cnn import ChestPneumoniaCNN, _build_eval_transform, resolve_data_root  # noqa: E402


def _load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    class_names = list(payload["class_names"])
    image_size = int(payload["image_size"])
    model = ChestPneumoniaCNN(num_classes=int(payload["num_classes"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, class_names, image_size


@torch.no_grad()
def evaluate_test_split(checkpoint: Path, data_root: Path, device: torch.device) -> dict:
    model, class_names, image_size = _load_model(checkpoint, device)
    test_ds = datasets.ImageFolder(str(data_root / "test"), transform=_build_eval_transform(image_size))
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    all_preds, all_targets, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_preds.extend(preds.tolist())
        all_targets.extend(labels.tolist())
        all_probs.extend(probs.tolist())

    acc = accuracy_score(all_targets, all_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=list(range(len(class_names))), average=None, zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(class_names))))
    confidences = [all_probs[i][all_preds[i]] for i in range(len(all_preds))]
    correct_mask = np.array(all_preds) == np.array(all_targets)

    return {
        "checkpoint": str(checkpoint),
        "image_size": image_size,
        "test_samples": len(all_targets),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "confusion_matrix": {"labels": class_names, "matrix": cm.tolist()},
        "per_class": {
            class_names[i]: {
                "precision": float(per_class_p[i]),
                "recall": float(per_class_r[i]),
                "f1": float(per_class_f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(class_names))
        },
        "confidence": {
            "mean_correct": float(np.mean(np.array(confidences)[correct_mask])) if correct_mask.any() else 0.0,
            "mean_incorrect": float(np.mean(np.array(confidences)[~correct_mask])) if (~correct_mask).any() else 0.0,
            "median_all": float(np.median(confidences)),
        },
        "misclassified_count": int((~correct_mask).sum()),
        "classification_report": classification_report(
            all_targets, all_preds, target_names=class_names, zero_division=0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=SCRIPT_DIR / "outputs" / "xray" / "xray_cnn.pt")
    parser.add_argument("--data-root", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs" / "xray" / "evaluation_report.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = resolve_data_root(args.data_root)
    inference = XRayInference(args.checkpoint, device=device)
    report = evaluate_test_split(args.checkpoint, data_root, device)

    samples = []
    for folder in (data_root / "test" / "NORMAL", data_root / "test" / "PNEUMONIA"):
        for path in sorted(folder.glob("*.*"))[:15]:
            out = inference.predict_image_path(path)
            samples.append(
                {
                    "file": path.name,
                    "true_class": folder.name,
                    "predicted_class": out["predicted_class"],
                    "correct": out["predicted_class"] == folder.name,
                    "confidence": round(out["confidence"], 4),
                    "probabilities": {k: round(v, 4) for k, v in out["probabilities"].items()},
                }
            )

    input_dir = SCRIPT_DIR.parent / "input"
    input_preds = []
    if input_dir.is_dir():
        for path in sorted(input_dir.glob("*.*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            out = inference.predict_image_path(path)
            input_preds.append(
                {
                    "file": path.name,
                    "predicted_class": out["predicted_class"],
                    "confidence": round(out["confidence"], 4),
                    "probabilities": {k: round(v, 4) for k, v in out["probabilities"].items()},
                }
            )

    report["sample_predictions"] = samples
    report["input_folder_predictions"] = input_preds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    labels = report["confusion_matrix"]["labels"]
    matrix = report["confusion_matrix"]["matrix"]
    print("=" * 60)
    print("CHEST X-RAY PNEUMONIA MODEL — FULL EVALUATION")
    print("=" * 60)
    print(f"Accuracy: {report['accuracy']:.4f} | Macro F1: {report['macro_f1']:.4f}")
    print(f"Misclassified: {report['misclassified_count']} / {report['test_samples']}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"{'':12} " + " ".join(f"{l:>12}" for l in labels))
    for i, row in enumerate(matrix):
        print(f"{labels[i]:12} " + " ".join(f"{v:>12}" for v in row))
    print("\nPer-class:")
    for name, m in report["per_class"].items():
        print(f"  {name:10} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
    print(f"\nConfidence correct={report['confidence']['mean_correct']:.3f} incorrect={report['confidence']['mean_incorrect']:.3f}")
    print(report["classification_report"])
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
