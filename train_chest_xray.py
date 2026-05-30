from __future__ import annotations

import csv
import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# #region agent log
_DEBUG_LOG_PATH = Path(__file__).resolve().parent / "debug-9f5fc4.log"


def _debug_log(location: str, message: str, data: dict, hypothesis_id: str, run_id: str = "pre-fix") -> None:
    import time

    payload = {
        "sessionId": "9f5fc4",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
    except OSError:
        pass


# #endregion


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in multi-label classification.
    
    Ref: Lin et al. "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) raw predictions
            targets: (N, C) binary target labels
        Returns:
            Scalar loss
        """
        sigmoid_p = torch.sigmoid(logits)
        
        # BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        
        # Focal loss modulation
        p_t = sigmoid_p * targets + (1 - sigmoid_p) * (1 - targets)
        modulation = (1 - p_t) ** self.gamma
        focal = self.alpha * modulation * bce_loss
        
        return focal.mean()


class ChestXRayCNN(nn.Module):
    """ResNet50 backbone with custom classifier for 14-class chest X-ray disease detection."""
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        # Load ResNet50 (pretrained if available, otherwise from scratch)
        try:
            if pretrained:
                from torchvision.models import ResNet50_Weights
                self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet50(weights=None)
        except Exception as e:
            # Fallback: load without pretrained weights if SSL/download fails
            print(f"Note: Could not load pretrained weights ({type(e).__name__}), using random initialization")
            self.backbone = models.resnet50(weights=None)
        
        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Custom classifier head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        
        # Store backbone for Grad-CAM
        self.features = self.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.classifier(x)


def _resolve_image_root(data_root: Path) -> Path:
    train_dir = data_root / "train"
    if train_dir.exists() and train_dir.is_dir():
        return train_dir
    return data_root


def _find_nih_image_root(data_root: Path) -> Path:
    for child in sorted(data_root.iterdir()):
        if child.is_dir() and child.name.startswith("images_"):
            return data_root
    return data_root


class NIHChestXrayDataset(Dataset):
    def __init__(self, data_root: Path, transform=None) -> None:
        self.data_root = data_root
        self.transform = transform
        self.image_root = _find_nih_image_root(data_root)
        self.image_paths = self._index_images()
        csv_path = data_root / "Data_Entry_2017.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing NIH label CSV: {csv_path}")
        # NIH 14 labels (multi-label)
        self.label_names = [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
        ]

        self.samples: list[tuple[Path, list[int]]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_name = row["Image Index"].strip()
                finding_labels = row["Finding Labels"].strip()
                multi = [0] * len(self.label_names)
                if finding_labels and finding_labels != "No Finding":
                    for lab in [s.strip() for s in finding_labels.split("|") if s.strip()]:
                        # CSV uses '|' as separator in some exports; support comma too
                        parts = [p.strip() for p in lab.replace(",", "|").split("|") if p.strip()]
                        for p in parts:
                            if p in self.label_names:
                                multi[self.label_names.index(p)] = 1
                image_path = self._locate_image(image_name)
                if image_path is not None:
                    self.samples.append((image_path, multi))

        if not self.samples:
            raise ValueError(f"No NIH images found under {data_root}")

        self.classes = list(self.label_names)

    def _index_images(self) -> dict[str, Path]:
        indexed_paths: dict[str, Path] = {}
        for pattern in ("images_*/images/*.png", "images_*/images/*.jpg", "images_*/images/*.jpeg"):
            for image_path in self.data_root.glob(pattern):
                indexed_paths.setdefault(image_path.name, image_path)
        return indexed_paths

    def _locate_image(self, image_name: str) -> Path | None:
        direct_path = self.data_root / image_name
        if direct_path.exists():
            return direct_path
        return self.image_paths.get(image_name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        # return multi-hot as float tensor for BCEWithLogitsLoss
        return image, torch.tensor(label, dtype=torch.float32)


def _limit_dataset(dataset: Dataset, limit: int, seed: int) -> Subset:
    limit = min(limit, len(dataset))
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return Subset(dataset, indices[:limit])


def _build_nih_subset(dataset: NIHChestXrayDataset, limit: int, seed: int) -> Subset:
    limit = min(limit, len(dataset))
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {0: [], 1: []}
    for index, (_, label) in enumerate(dataset.samples):
        # `label` can be a multi-hot list for NIH (unhashable).
        # Treat as positive if any label is present, else negative.
        if isinstance(label, (list, tuple, np.ndarray)):
            positive = 1 if any(label) else 0
        else:
            positive = int(label)
        by_label[positive].append(index)

    for label_indices in by_label.values():
        rng.shuffle(label_indices)

    half = limit // 2
    selected = by_label[0][:half] + by_label[1][: max(0, limit - half)]
    rng.shuffle(selected)
    # Ensure we don't exceed the requested limit
    selected = selected[:limit]
    return Subset(dataset, selected)


def _build_train_transform(image_size: int):
    """Aggressive augmentation for medical imaging with small-to-medium datasets."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            # Stronger spatial augmentations for robustness
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=10),
            transforms.RandomHorizontalFlip(p=0.5),
            # Intensity augmentations
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomInvert(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            # Random erasing for occlusion robustness
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.0)),
        ]
    )


def _build_eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _compute_pos_weight_for_subset(dataset: NIHChestXrayDataset, subset: Subset) -> torch.Tensor:
    num_labels = len(dataset.label_names)
    positives = [0] * num_labels
    subset_size = len(subset)

    for index in subset.indices:
        _, label = dataset.samples[index]
        for label_index, value in enumerate(label):
            positives[label_index] += int(value)

    weights = []
    for positive_count in positives:
        negative_count = subset_size - positive_count
        weights.append(float(negative_count / max(1, positive_count)) if positive_count > 0 else 1.0)

    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    data_root: Path,
    batch_size: int,
    max_images: int,
    image_size: int,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], torch.Tensor | None]:
    train_transform = _build_train_transform(image_size)
    eval_transform = _build_eval_transform(image_size)

    csv_path = data_root / "Data_Entry_2017.csv"
    if csv_path.exists():
        train_dataset_source = NIHChestXrayDataset(data_root=data_root, transform=train_transform)
        eval_dataset_source = NIHChestXrayDataset(data_root=data_root, transform=eval_transform)
        if max_images and max_images > 0:
            limited_dataset = _build_nih_subset(train_dataset_source, max_images, seed)
        else:
            limited_dataset = Subset(train_dataset_source, list(range(len(train_dataset_source))))
        class_names = train_dataset_source.classes
    else:
        image_root = _resolve_image_root(data_root)
        train_dataset_source = datasets.ImageFolder(root=str(image_root), transform=train_transform)
        eval_dataset_source = datasets.ImageFolder(root=str(image_root), transform=eval_transform)
        if len(train_dataset_source) == 0:
            raise ValueError(f"No images found under {image_root}")
        if max_images and max_images > 0:
            limited_dataset = _limit_dataset(train_dataset_source, max_images, seed)
        else:
            limited_dataset = Subset(train_dataset_source, list(range(len(train_dataset_source))))
        class_names = list(train_dataset_source.classes)

    if len(limited_dataset) < 3:
        raise ValueError("Need at least 3 images after subsampling to create train/validation/test splits.")

    total_size = len(limited_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    if train_size == 0:
        train_size = 1
    if val_size == 0:
        val_size = 1
    if test_size == 0:
        test_size = 1

    while train_size + val_size + test_size > total_size:
        if train_size >= val_size and train_size >= test_size and train_size > 1:
            train_size -= 1
        elif val_size >= test_size and val_size > 1:
            val_size -= 1
        elif test_size > 1:
            test_size -= 1
        else:
            break

    remainder = total_size - (train_size + val_size + test_size)
    train_size += remainder

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        limited_dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    base_indices = list(limited_dataset.indices)
    train_indices = [base_indices[index] for index in train_dataset.indices]
    val_indices = [base_indices[index] for index in val_dataset.indices]
    test_indices = [base_indices[index] for index in test_dataset.indices]

    train_dataset = Subset(train_dataset_source, train_indices)
    val_dataset = Subset(eval_dataset_source, val_indices)
    test_dataset = Subset(eval_dataset_source, test_indices)

    pos_weight = _compute_pos_weight_for_subset(train_dataset_source, train_dataset) if csv_path.exists() else None

    train_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": max(0, num_workers),
    }
    eval_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": max(0, num_workers),
    }
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 2
        eval_loader_kwargs["persistent_workers"] = True
        eval_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, **eval_loader_kwargs)
    return train_loader, val_loader, test_loader, class_names, pos_weight


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int = 50,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    batch_count = 0

    print(f"  Training batches: {len(loader)}")

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        batch_count = batch_index

        if log_interval > 0 and batch_index % log_interval == 0:
            print(f"    batch {batch_index}/{len(loader)} | avg_loss={running_loss / max(1, total):.4f}")

    if batch_count and batch_count % log_interval != 0:
        print(f"    batch {batch_count}/{len(loader)} | avg_loss={running_loss / max(1, total):.4f}")

    return running_loss / max(1, total)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    total = 0
    print(f"  Validation batches: {len(loader)}")
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    return running_loss / max(1, total)


def collect_test_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int], list[float]]:
    model.eval()
    running_loss = 0.0
    total = 0
    targets: list[list[int]] = []
    predictions: list[list[int]] = []
    probabilities_list: list[list[float]] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).int()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        targets.extend(labels.cpu().tolist())
        predictions.extend(preds.cpu().tolist())
        probabilities_list.extend(probs.cpu().tolist())

    return running_loss / max(1, total), targets, predictions, probabilities_list


def _prediction_type(prediction: int, label: int) -> str:
    if prediction == 1 and label == 1:
        return "True Positive"
    if prediction == 1 and label == 0:
        return "False Positive"
    if prediction == 0 and label == 1:
        return "False Negative"
    return "True Negative"


def _binary_confusion_stats(targets: list[int], predictions: list[int]) -> dict[str, float | int]:
    true_positive = sum(1 for target, prediction in zip(targets, predictions) if target == 1 and prediction == 1)
    true_negative = sum(1 for target, prediction in zip(targets, predictions) if target == 0 and prediction == 0)
    false_positive = sum(1 for target, prediction in zip(targets, predictions) if target == 0 and prediction == 1)
    false_negative = sum(1 for target, prediction in zip(targets, predictions) if target == 1 and prediction == 0)

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1_score = 2 * precision * recall / max(1e-8, precision + recall)
    accuracy = (true_positive + true_negative) / max(1, len(targets))

    return {
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
    }


def _format_confidence(probability: float) -> str:
    return f"{probability:.2f}"


def _get_last_conv_layer(model: nn.Module) -> nn.Module:
    # Robustly find the last Conv2d in the model (works for sequential features
    # and for models like torchvision ResNet where `features` is not iterable).
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No convolutional layer found for Grad-CAM.")
    return last_conv


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.target_layer = target_layer
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, outputs) -> None:
        self.activations = outputs.detach()

    def _save_gradients(self, _module, _grad_inputs, grad_outputs) -> None:
        self.gradients = grad_outputs[0].detach()

    def __call__(self, inputs: torch.Tensor, class_index: int) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(inputs)
        target_score = outputs[:, class_index].sum()
        target_score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def _make_gradcam_overlay(image: Image.Image, heatmap: torch.Tensor, alpha: float = 0.4) -> Image.Image:
    heatmap_np = heatmap.detach().cpu().numpy()
    heatmap_colored = colormaps["jet"](heatmap_np)[..., :3]
    heatmap_image = Image.fromarray((heatmap_colored * 255).astype(np.uint8)).resize(image.size)
    base_image = image.convert("RGB")
    return Image.blend(base_image, heatmap_image, alpha=alpha)


def _resolve_sample_path(dataset, index: int) -> Path | None:
    if isinstance(dataset, Subset):
        return _resolve_sample_path(dataset.dataset, dataset.indices[index])

    samples = getattr(dataset, "samples", None)
    if samples is not None and 0 <= index < len(samples):
        return Path(samples[index][0])

    return None


def _resolve_sample_image(dataset, index: int) -> Image.Image:
    sample_path = _resolve_sample_path(dataset, index)
    if sample_path is None:
        raise FileNotFoundError("Could not resolve source image for preview saving.")
    return Image.open(sample_path).convert("RGB")


def _save_preview_image(
    image: Image.Image,
    gradcam_overlay: Image.Image,
    output_path: Path,
    prediction_text: str,
    true_text: str,
    result_text: str,
    confidence_text: str,
    extra_summary_lines: list[str] | None = None,
) -> None:
    preview = image.convert("RGB")
    gradcam_preview = gradcam_overlay.convert("RGB")
    text_height = 110
    canvas_width = preview.width + gradcam_preview.width
    canvas_height = max(preview.height, gradcam_preview.height) + text_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    canvas.paste(preview, (0, 0))
    canvas.paste(gradcam_preview, (preview.width, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    lines = [
        f"Prediction: {prediction_text}",
        f"Confidence: {confidence_text}",
        f"True: {true_text}",
        f"Type: {result_text}",
    ]
    if extra_summary_lines:
        lines.extend(extra_summary_lines)
    y = max(preview.height, gradcam_preview.height) + 8
    for line in lines:
        draw.text((8, y), line, fill=(0, 0, 0), font=font)
        y += 20
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def preview_test_predictions(
    model: nn.Module,
    dataset,
    class_names: list[str],
    device: torch.device,
    output_dir: Path,
    test_predictions: list[int],
    test_confidences: list[float],
    max_samples: int = 5,
    decision_thresholds: dict[str, float] | None = None,
) -> None:
    decision_thresholds = decision_thresholds or {name: 0.5 for name in class_names}
    # Multi-label preview: show threshold-based predicted diseases with probabilities
    preview_count = min(max_samples, len(dataset))
    print(f"Previewing first {preview_count} test samples:")
    gradcam = GradCAM(model, _get_last_conv_layer(model))

    for index in range(preview_count):
        image, labels = dataset[index]
        input_tensor = image.unsqueeze(0).to(device)
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

        display_text, positive_labels, gradcam_index, headline_conf = _format_clinical_prediction(
            probs, class_names, decision_thresholds
        )
        labelled_sorted = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)
        topk = labelled_sorted[:3]
        interpretation_lines = []
        if positive_labels:
            for label in positive_labels:
                idx = class_names.index(label)
                interpretation_lines.append(f"Possible signs of {label} detected ({probs[idx]:.2f})")
        elif headline_conf >= 0.2:
            interpretation_lines.append(
                f"No disease above threshold; highest raw score {class_names[gradcam_index]} ({headline_conf:.2f})"
            )
        else:
            interpretation_lines = ["No strong signs detected (low confidence)"]

        sample_path = _resolve_sample_path(dataset, index)
        path_text = sample_path.name if sample_path is not None else f"sample_{index + 1}"
        print(f"Test {index + 1}: {path_text} | Prediction: {display_text}")
        for line in interpretation_lines:
            print("  -", line)

        source_image = _resolve_sample_image(dataset, index)
        heatmap = gradcam(input_tensor, int(gradcam_index))
        gradcam_overlay = _make_gradcam_overlay(source_image, heatmap)
        preview_name = f"test_preview_{index + 1}_{path_text}"
        preview_path = output_dir / "test_previews" / preview_name
        _save_preview_image(
            source_image,
            gradcam_overlay,
            preview_path,
            display_text,
            "N/A",
            "; ".join(interpretation_lines),
            f"{headline_conf:.2f}",
        )
        print(f"Saved preview image: {preview_path}")

    gradcam.close()


def optimize_decision_thresholds(
    model: nn.Module,
    val_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> dict[str, float]:
    """Optimize per-class decision thresholds on validation set to maximize F1 score."""
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    
    thresholds = {}
    for label_idx in range(len(class_names)):
        # Grid search for optimal threshold for this label
        best_f1 = -1.0
        best_threshold = 0.5
        
        label_probs = all_probs[:, label_idx]
        label_targets = all_targets[:, label_idx]
        
        for threshold in np.arange(0.1, 0.95, 0.05):
            preds = (label_probs >= threshold).astype(int)
            tp = ((preds == 1) & (label_targets == 1)).sum()
            fp = ((preds == 1) & (label_targets == 0)).sum()
            fn = ((preds == 0) & (label_targets == 1)).sum()
            
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-8, precision + recall)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        thresholds[class_names[label_idx]] = float(best_threshold)

    # #region agent log
    _debug_log(
        "train_chest_xray.py:optimize_decision_thresholds",
        "validation thresholds computed",
        {
            "threshold_count": len(thresholds),
            "sample_thresholds": dict(list(thresholds.items())[:4]),
            "default_0_5_count": sum(1 for value in thresholds.values() if abs(value - 0.5) < 1e-6),
        },
        "H3",
    )
    # #endregion

    print(f"\nOptimized per-class thresholds (based on validation F1):")
    for name, thresh in thresholds.items():
        print(f"  {name}: {thresh:.3f}")
    
    return thresholds


def _load_checkpoint_payload(checkpoint_path: Path, device: torch.device) -> dict:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _resolve_checkpoint_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path
    best_path = checkpoint_path.parent / "chest_xray_cnn_best.pt"
    if best_path.exists():
        print(f"Note: using best checkpoint {best_path} ({checkpoint_path} not found)")
        return best_path
    return checkpoint_path


def _load_thresholds_for_checkpoint(checkpoint_path: Path, class_names: list[str]) -> dict[str, float]:
    resolved = _resolve_checkpoint_path(checkpoint_path)
    if resolved.exists():
        try:
            payload = _load_checkpoint_payload(resolved, torch.device("cpu"))
            thresholds = payload.get("optimized_thresholds")
            if isinstance(thresholds, dict):
                return {name: float(thresholds.get(name, 0.5)) for name in class_names}
        except Exception:
            pass

    metrics_path = checkpoint_path.parent / "test_metrics.json"
    if not metrics_path.exists():
        return {name: 0.5 for name in class_names}

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {name: 0.5 for name in class_names}

    thresholds = payload.get("optimized_thresholds")
    if not isinstance(thresholds, dict):
        return {name: 0.5 for name in class_names}

    return {name: float(thresholds.get(name, 0.5)) for name in class_names}


def _thresholded_prediction_vector(
    probs: list[float],
    class_names: list[str],
    decision_thresholds: dict[str, float],
) -> list[int]:
    return [
        1 if probs[class_idx] >= decision_thresholds.get(class_names[class_idx], 0.5) else 0
        for class_idx in range(len(class_names))
    ]


def _format_clinical_prediction(
    probs: list[float],
    class_names: list[str],
    decision_thresholds: dict[str, float],
) -> tuple[str, list[str], int, float]:
    """Return display text, positive labels, Grad-CAM class index, and headline confidence."""
    thresholded = _thresholded_prediction_vector(probs, class_names, decision_thresholds)
    positive_indices = [idx for idx, active in enumerate(thresholded) if active == 1]
    ranked = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)

    if positive_indices:
        positive_indices.sort(key=lambda idx: probs[idx], reverse=True)
        positive_labels = [class_names[idx] for idx in positive_indices]
        display_text = ", ".join(f"{label} ({probs[idx]:.2f})" for idx, label in zip(positive_indices, positive_labels))
        gradcam_index = positive_indices[0]
        headline_conf = probs[gradcam_index]
        return display_text, positive_labels, gradcam_index, headline_conf

    top_idx, top_prob = ranked[0]
    if top_prob >= 0.2:
        display_text = f"No Finding (weak signal: {class_names[top_idx]} {top_prob:.2f})"
    else:
        display_text = "No Finding (low confidence)"
    return display_text, [], top_idx, top_prob


def save_checkpoint(
    model: nn.Module,
    class_names: Iterable[str],
    output_dir: Path,
    optimized_thresholds: dict[str, float] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "chest_xray_cnn.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "class_names": list(class_names),
    }
    if optimized_thresholds is not None:
        payload["optimized_thresholds"] = optimized_thresholds
    torch.save(payload, checkpoint_path)
    metadata_path = output_dir / "chest_xray_classes.json"
    metadata_path.write_text(json.dumps({"class_names": list(class_names)}, indent=2))
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet50-based chest X-ray disease classifier with Focal Loss.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "Downloads" / "archive",
        help="Path to the chest X-ray dataset folder (default: ~/Downloads/archive).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum images to use for training (default: 0 = use full dataset). For testing, use 200-500.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Resize images to this square size.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training and validation (reduced to 8 for better gradient signals).")
    parser.add_argument("--num-workers", type=int, default=6, help="DataLoader worker processes for faster CPU/GPU input pipeline.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100 for convergence with large datasets).")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate (reduced for pretrained backbone).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory to save checkpoints.")
    parser.add_argument(
        "--external-test-folder",
        type=Path,
        default=Path("input"),
        help="Path to a folder of external images to run inference on after training (default: input/).",
    )
    parser.add_argument(
        "--external-labels-csv",
        type=Path,
        default=None,
        help="Optional CSV with labels for `--external-test-folder` images. Must contain filename and Finding Labels columns.",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Run model inference on `--external-test-folder` using a saved checkpoint without training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs") / "chest_xray_cnn.pt",
        help="Path to a saved model checkpoint to load for inference-only mode (default: outputs/chest_xray_cnn.pt).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--use-focal-loss", action="store_true", default=True, help="Use Focal Loss for class imbalance (default: True).")
    return parser.parse_args()


def _load_external_images(folder: Path, image_size: int):
    if folder is None:
        return []
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"External test folder does not exist: {folder}")

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    images: list[tuple[Path, torch.Tensor]] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in sorted(folder.glob(ext)):
            try:
                img = Image.open(p).convert("RGB")
                tensor = transform(img)
                images.append((p, tensor))
            except Exception:
                continue
    return images


def _parse_multi_label_text(label_text: str, class_names: list[str]) -> list[int]:
    multi = [0] * len(class_names)
    cleaned_text = (label_text or "").strip()
    if not cleaned_text or cleaned_text == "No Finding":
        return multi

    for raw_label in cleaned_text.replace(",", "|").split("|"):
        label = raw_label.strip()
        if label and label in class_names:
            multi[class_names.index(label)] = 1
    return multi


def _load_external_labels_csv(labels_csv: Path, class_names: list[str]) -> dict[str, list[int]]:
    if not labels_csv.exists():
        raise FileNotFoundError(f"External labels CSV not found: {labels_csv}")

    filename_keys = ("filename", "image", "image_index", "image name")
    label_keys = ("finding labels", "labels", "label", "finding_label")
    label_map: dict[str, list[int]] = {}

    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"External labels CSV has no header row: {labels_csv}")

        normalized = {name.lower().strip(): name for name in reader.fieldnames}
        filename_column = next((normalized[key] for key in filename_keys if key in normalized), None)
        label_column = next((normalized[key] for key in label_keys if key in normalized), None)

        if filename_column is None or label_column is None:
            raise ValueError(
                "External labels CSV must contain filename and labels columns. "
                "Accepted examples: filename + Finding Labels."
            )

        for row in reader:
            filename = Path(row[filename_column].strip()).name
            label_map[filename] = _parse_multi_label_text(row[label_column], class_names)

    return label_map


def run_external_test_inference(
    model: nn.Module,
    images: list[tuple[Path, torch.Tensor]],
    class_names: list[str],
    device: torch.device,
    output_dir: Path,
    image_size: int,
    external_targets: dict[str, list[int]] | None = None,
    decision_thresholds: dict[str, float] | None = None,
):
    if not images:
        print("No external images found to run inference on.")
        return

    model.eval()
    gradcam = GradCAM(model, _get_last_conv_layer(model))
    output_dir = output_dir / "external_test_previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_thresholds = decision_thresholds or {name: 0.5 for name in class_names}

    results = []
    all_targets: list[list[int]] = []
    all_predictions: list[list[int]] = []
    all_probabilities: list[list[float]] = []
    for idx, (path, tensor) in enumerate(images, start=1):
        input_tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
            labelled_sorted = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)
            thresholded_prediction = _thresholded_prediction_vector(probs, class_names, decision_thresholds)
            display_text, positive_labels, gradcam_index, headline_conf = _format_clinical_prediction(
                probs, class_names, decision_thresholds
            )

        # #region agent log
        _debug_log(
            "train_chest_xray.py:run_external_test_inference",
            "external prediction breakdown",
            {
                "filename": path.name,
                "top1_label": class_names[labelled_sorted[0][0]],
                "top1_prob": round(float(labelled_sorted[0][1]), 4),
                "display_prediction": display_text,
                "threshold_positives": positive_labels,
                "thresholds_source": "custom" if any(abs(v - 0.5) > 1e-6 for v in decision_thresholds.values()) else "all_default_0.5",
                "display_uses_top1_not_threshold": False,
            },
            "H1",
            run_id="post-fix",
        )
        # #endregion

        interpretation = []
        if positive_labels:
            interpretation = [f"Threshold-positive findings: {', '.join(positive_labels)}"]
        elif headline_conf >= 0.2:
            interpretation = [f"No disease above threshold; highest raw score {class_names[gradcam_index]} ({headline_conf:.2f})"]
        else:
            interpretation = ["No strong signs detected (low confidence)"]
        target = external_targets.get(path.name) if external_targets else None
        if target is not None:
            all_targets.append(target)
            all_predictions.append(thresholded_prediction)
            all_probabilities.append(probs)
        summary_lines = [
            f"Predicted: {display_text}",
            f"Headline confidence: {headline_conf:.2f}",
        ]
        print(f"External {idx}: {path.name} | {display_text}")
        for line in interpretation:
            print("  -", line)

        source_image = Image.open(path).convert("RGB")
        heatmap = gradcam(input_tensor, int(gradcam_index))
        gradcam_overlay = _make_gradcam_overlay(source_image, heatmap)

        preview_name = f"external_preview_{idx}_{path.name}"
        preview_path = output_dir / preview_name
        _save_preview_image(
            source_image,
            gradcam_overlay,
            preview_path,
            display_text,
            "Input image",
            "External inference",
            f"{headline_conf:.2f}",
            summary_lines,
        )
        print(f"Saved external preview: {preview_path}")
        row = {"filename": path.name, "preview": str(preview_path)}
        row["predicted_label"] = ", ".join(positive_labels) if positive_labels else "No Finding"
        row["predicted_confidence"] = float(headline_conf)
        row["top1"] = display_text
        if target is not None:
            row["ground_truth"] = "|".join([class_names[i] for i, value in enumerate(target) if value == 1]) or "No Finding"
        for i, name in enumerate(class_names):
            row[name] = probs[i]
        row["interpretation"] = " | ".join(interpretation)
        results.append(row)

    gradcam.close()
    # write CSV summary
    csv_path = output_dir / "external_test_results.csv"
    try:
        # write CSV with per-label probabilities
        fieldnames = ["filename", "predicted_label", "predicted_confidence", "top1"]
        if external_targets:
            fieldnames.append("ground_truth")
        fieldnames += list(class_names) + ["preview", "interpretation"]
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Wrote external results CSV: {csv_path}")
    except Exception as exc:
        print(f"Failed to write external results CSV: {exc}")

    if all_targets:
        targets_np = np.array(all_targets)
        preds_np = np.array(all_predictions)
        probs_np = np.array(all_probabilities)
        per_label_precision, per_label_recall, per_label_f1, _ = precision_recall_fscore_support(
            targets_np, preds_np, average=None, zero_division=0
        )
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            targets_np, preds_np, average="micro", zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            targets_np, preds_np, average="macro", zero_division=0
        )
        subset_acc = accuracy_score(targets_np, preds_np)

        metrics = {
            "subset_accuracy": float(subset_acc),
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "micro_f1": float(micro_f1),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "decision_thresholds": decision_thresholds,
            "per_label": {},
        }
        for i, name in enumerate(class_names):
            metrics["per_label"][name] = {
                "precision": float(per_label_precision[i]),
                "recall": float(per_label_recall[i]),
                "f1": float(per_label_f1[i]),
                "threshold": float(decision_thresholds.get(name, 0.5)),
                "prevalence": int(targets_np[:, i].sum()),
            }

        metrics_path = output_dir / "external_test_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"External labeled metrics | subset_accuracy={subset_acc:.4f} | micro_f1={micro_f1:.4f} | macro_f1={macro_f1:.4f}")
        print(f"Saved external metrics to {metrics_path}")

        # save per-image JSON summaries for FusionNet
        try:
            for row in results:
                fname = Path(row["preview"]).name
                json_name = (output_dir / (Path(fname).stem + ".json"))
                # reconstruct probs and interpretation from row
                probs = [float(row[name]) for name in class_names]
                fusion = format_for_fusionnet(probs, class_names)
                summary = {"filename": row["filename"], "preview": row["preview"], "probabilities": {name: float(row[name]) for name in class_names}, "interpretation": fusion["interpretation"], "combined_text": fusion["text"]}
                json_name.parent.mkdir(parents=True, exist_ok=True)
                with json_name.open("w", encoding="utf-8") as jf:
                    json.dump(summary, jf, indent=2)
            print(f"Wrote per-image JSON summaries to {output_dir}")
        except Exception as exc:
            print(f"Failed to write per-image JSON summaries: {exc}")


def format_for_fusionnet(probs: list[float], class_names: list[str], top_k: int = 6) -> dict:
    """Create a combined diagnosis block for FusionNet ingestion.

    Returns a dict with a human-readable text block, a ranked list of (label, prob),
    and conservative interpretation lines.
    """
    ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    ranked_top = ranked[:top_k]
    diagnosis_lines = ["Combined Diagnosis:"]
    for idx, p in ranked_top:
        diagnosis_lines.append(f"- {class_names[idx]}: {p:.2f}")

    interpretation = []
    for idx, p in ranked_top:
        if p >= 0.5:
            interpretation.append(f"Possible signs of {class_names[idx]} detected ({p:.2f})")
        elif p >= 0.2:
            interpretation.append(f"Weak evidence of {class_names[idx]} ({p:.2f})")
    if not interpretation:
        interpretation = ["No strong signs detected (low confidence)"]

    text_block = "\n".join(diagnosis_lines) + "\n\nInterpretation:\n" + "\n".join(f"- {s}" for s in interpretation)
    return {
        "text": text_block,
        "ranked": [(class_names[idx], float(p)) for idx, p in ranked_top],
        "interpretation": interpretation,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Inference-only mode: load checkpoint and run external-folder inference
    if getattr(args, "inference_only", False):
        if args.external_test_folder is None:
            raise ValueError("--inference-only requires --external-test-folder to be set.")
        checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
        best_checkpoint_path = checkpoint_path.parent / "chest_xray_cnn_best.pt"
        metrics_path = checkpoint_path.parent / "test_metrics.json"
        # #region agent log
        _debug_log(
            "train_chest_xray.py:main:inference_only",
            "inference-only checkpoint context",
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_exists": checkpoint_path.exists(),
                "best_checkpoint_exists": best_checkpoint_path.exists(),
                "metrics_json_exists": metrics_path.exists(),
                "external_folder": str(args.external_test_folder),
            },
            "H4",
            run_id="post-fix",
        )
        # #endregion
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        payload = _load_checkpoint_payload(checkpoint_path, device)
        class_names = payload.get("class_names")
        if class_names is None:
            raise ValueError("Checkpoint missing class_names metadata.")
        model = ChestXRayCNN(num_classes=len(class_names), pretrained=False).to(device)
        model.load_state_dict(payload["model_state_dict"])
        external_images = _load_external_images(args.external_test_folder, args.image_size)
        external_targets = None
        if args.external_labels_csv is not None:
            external_targets = _load_external_labels_csv(args.external_labels_csv, class_names)
        decision_thresholds = _load_thresholds_for_checkpoint(checkpoint_path, class_names)
        # #region agent log
        _debug_log(
            "train_chest_xray.py:main:inference_only",
            "loaded decision thresholds",
            {
                "threshold_sample": dict(list(decision_thresholds.items())[:4]),
                "using_defaults_only": all(abs(v - 0.5) < 1e-6 for v in decision_thresholds.values()),
            },
            "H3",
            run_id="post-fix",
        )
        # #endregion
        run_external_test_inference(
            model,
            external_images,
            class_names,
            device,
            args.output_dir,
            args.image_size,
            external_targets=external_targets,
            decision_thresholds=decision_thresholds,
        )
        return
    
    train_loader, val_loader, test_loader, class_names, pos_weight = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        max_images=args.max_images,
        image_size=args.image_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = ChestXRayCNN(num_classes=len(class_names), pretrained=True).to(device)
    multi_label = len(class_names) > 2
    
    # Use Focal Loss for better handling of class imbalance
    if getattr(args, "use_focal_loss", True) and multi_label:
        criterion = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight.to(device) if pos_weight is not None else None)
        print("Using Focal Loss with gamma=2.0 for class imbalance handling")
    elif multi_label and pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        print("Using weighted BCEWithLogitsLoss")
    elif multi_label:
        criterion = nn.BCEWithLogitsLoss()
        print("Using BCEWithLogitsLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss (single-label mode)")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )

    print(f"\nTraining Configuration:")
    print(f"  Dataset root: {args.data_root}")
    print(f"  Classes: {len(class_names)} ({', '.join(class_names[:3])}...)")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  Batch size: {args.batch_size} (smaller for better gradients on large datasets)")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    if multi_label and pos_weight is not None:
        print(f"  Class imbalance weights: enabled")

    print(f"\nStarting training with ResNet50 backbone and Focal Loss...\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10

    for epoch in range(1, args.epochs + 1):
        epoch_start = __import__("time").time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        epoch_seconds = __import__("time").time() - epoch_start
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={epoch_seconds:.1f}s"
        )
        
        # Early stopping with patience
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best checkpoint
            checkpoint_path = args.output_dir / "chest_xray_cnn_best.pt"
            args.output_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
            }
            torch.save(payload, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (val_loss not improving)")
                break

    best_ckpt_path = args.output_dir / "chest_xray_cnn_best.pt"
    reloaded_best = False
    if best_ckpt_path.exists():
        best_payload = _load_checkpoint_payload(best_ckpt_path, device)
        model.load_state_dict(best_payload["model_state_dict"])
        reloaded_best = True
        print(f"Restored best-validation checkpoint from {best_ckpt_path}")

    final_weight_probe = float(next(model.parameters()).detach().cpu().mean().item())
    # #region agent log
    _debug_log(
        "train_chest_xray.py:main:post_train",
        "weights after optional best-checkpoint reload",
        {
            "epoch_stopped": epoch,
            "best_val_loss": float(best_val_loss),
            "patience_counter": patience_counter,
            "best_checkpoint_exists": best_ckpt_path.exists(),
            "reloaded_best": reloaded_best,
            "model_weight_mean_probe": round(final_weight_probe, 8),
        },
        "H2",
        run_id="post-fix",
    )
    # #endregion

    optimized_thresholds = optimize_decision_thresholds(model, val_loader, class_names, device)

    model.eval()
    test_targets: list[list[int]] = []
    test_predictions: list[list[int]] = []
    test_probabilities: list[list[float]] = []
    test_loss_running = 0.0
    test_loss_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            preds = torch.zeros_like(probs)
            for class_idx, class_name in enumerate(class_names):
                threshold = optimized_thresholds.get(class_name, 0.5)
                preds[:, class_idx] = (probs[:, class_idx] >= threshold).float()

            test_loss_running += loss.item() * images.size(0)
            test_loss_total += labels.size(0)
            test_targets.extend(labels.cpu().tolist())
            test_predictions.extend(preds.cpu().tolist())
            test_probabilities.extend(probs.cpu().tolist())

    test_loss = test_loss_running / max(1, test_loss_total)
    print(f"\nTest set | test_loss={test_loss:.4f}")

    # compute multi-label metrics
    targets_np = np.array(test_targets)
    preds_np = np.array(test_predictions)
    probs_np = np.array(test_probabilities)

    if multi_label:
        per_label_precision, per_label_recall, per_label_f1, _ = precision_recall_fscore_support(
            targets_np, preds_np, average=None, zero_division=0
        )
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(targets_np, preds_np, average="micro", zero_division=0)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(targets_np, preds_np, average="macro", zero_division=0)
        subset_acc = accuracy_score(targets_np, preds_np)

        metrics = {
            "test_loss": test_loss,
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "micro_f1": float(micro_f1),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "subset_accuracy": float(subset_acc),
            "per_label": {},
            "optimized_thresholds": optimized_thresholds,
        }
        for i, name in enumerate(class_names):
            metrics["per_label"][name] = {
                "precision": float(per_label_precision[i]),
                "recall": float(per_label_recall[i]),
                "f1": float(per_label_f1[i]),
                "threshold": optimized_thresholds.get(name, 0.5),
                "prevalence": int(targets_np[:, i].sum()),
            }
        
        print(f"\n{'='*70}")
        print(f"FINAL TEST METRICS (with optimized per-class thresholds)")
        print(f"{'='*70}")
        print(f"Micro Precision: {micro_p:.4f} | Micro Recall: {micro_r:.4f} | Micro F1: {micro_f1:.4f}")
        print(f"Macro Precision: {macro_p:.4f} | Macro Recall: {macro_r:.4f} | Macro F1: {macro_f1:.4f}")
        print(f"Subset Accuracy: {subset_acc:.4f}")
        print(f"{'='*70}\n")
    else:
        # fallback single-label metrics
        flat_targets = targets_np.tolist()
        flat_preds = preds_np.tolist()
        acc = accuracy_score(flat_targets, flat_preds)
        p, r, f1, _ = precision_recall_fscore_support(flat_targets, flat_preds, average="binary", zero_division=0)
        metrics = {"test_loss": test_loss, "accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)}

    metrics_path = args.output_dir / "test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved test metrics to {metrics_path}")

    preview_test_predictions(
        model,
        test_loader.dataset,
        class_names,
        device,
        args.output_dir,
        [],
        [],
        max_samples=5,
        decision_thresholds=optimized_thresholds,
    )

    checkpoint_path = save_checkpoint(
        model,
        class_names,
        args.output_dir,
        optimized_thresholds=optimized_thresholds if multi_label else None,
    )
    if best_ckpt_path.exists():
        best_payload = {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
        }
        if multi_label:
            best_payload["optimized_thresholds"] = optimized_thresholds
        torch.save(best_payload, best_ckpt_path)
    # #region agent log
    if best_ckpt_path.exists():
        try:
            best_payload = torch.load(best_ckpt_path, map_location="cpu")
            best_probe = float(next(iter(best_payload["model_state_dict"].values())).mean().item())
            final_probe = float(next(model.parameters()).detach().cpu().mean().item())
            _debug_log(
                "train_chest_xray.py:main:save_checkpoint",
                "final vs best checkpoint weight comparison",
                {
                    "final_checkpoint": str(checkpoint_path),
                    "best_checkpoint": str(best_ckpt_path),
                    "best_weight_probe": round(best_probe, 8),
                    "final_weight_probe": round(final_probe, 8),
                    "weights_match_best": abs(best_probe - final_probe) < 1e-6,
                    "thresholds_saved_in_checkpoint": isinstance(best_payload.get("optimized_thresholds"), dict),
                },
                "H2",
                run_id="post-fix",
            )
        except Exception as exc:
            _debug_log(
                "train_chest_xray.py:main:save_checkpoint",
                "failed to compare best vs final checkpoint",
                {"error": type(exc).__name__},
                "H2",
                run_id="post-fix",
            )
    # #endregion
    print(f"Saved final checkpoint to {checkpoint_path}")

    # Run inference on the external input folder after training.
    if getattr(args, "external_test_folder", None) is not None:
        ext_folder = args.external_test_folder
        external_images = _load_external_images(ext_folder, args.image_size)
        external_targets = None
        if args.external_labels_csv is not None:
            external_targets = _load_external_labels_csv(args.external_labels_csv, class_names)
        decision_thresholds = _load_thresholds_for_checkpoint(checkpoint_path, class_names)
        run_external_test_inference(
            model,
            external_images,
            class_names,
            device,
            args.output_dir,
            args.image_size,
            external_targets=external_targets,
            decision_thresholds=decision_thresholds,
        )


if __name__ == "__main__":
    main()
