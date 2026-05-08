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
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class ChestXRayCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
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
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(7),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
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
    for layer in reversed(list(model.features)):
        if isinstance(layer, nn.Conv2d):
            return layer
    raise ValueError("No convolutional layer found for Grad-CAM.")


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
) -> None:
    # Multi-label preview: show top predicted diseases with probabilities and conservative wording
    preview_count = min(max_samples, len(dataset))
    print(f"Previewing first {preview_count} test samples:")
    gradcam = GradCAM(model, _get_last_conv_layer(model))

    for index in range(preview_count):
        image, labels = dataset[index]
        input_tensor = image.unsqueeze(0).to(device)
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

        # get top-3 disease predictions
        labelled = list(enumerate(probs))
        labelled_sorted = sorted(labelled, key=lambda x: x[1], reverse=True)
        topk = labelled_sorted[:3]
        top_text = ", ".join([f"{class_names[i]} ({p:.2f})" for i, p in topk if p >= 0.01])
        interpretation_lines = []
        for i, p in topk:
            if p >= 0.5:
                interpretation_lines.append(f"Possible signs of {class_names[i]} detected ({p:.2f})")
            elif p >= 0.2:
                interpretation_lines.append(f"Weak evidence of {class_names[i]} ({p:.2f})")
        if not interpretation_lines:
            interpretation_lines = ["No strong signs detected (low confidence)"]

        sample_path = _resolve_sample_path(dataset, index)
        path_text = sample_path.name if sample_path is not None else f"sample_{index + 1}"
        print(f"Test {index + 1}: {path_text} | Top: {top_text}")
        for line in interpretation_lines:
            print("  -", line)

        source_image = _resolve_sample_image(dataset, index)
        # use highest-confidence label to generate Grad-CAM visualization
        highest_label = labelled_sorted[0][0]
        heatmap = gradcam(input_tensor, int(highest_label))
        gradcam_overlay = _make_gradcam_overlay(source_image, heatmap)
        preview_name = f"test_preview_{index + 1}_{path_text}"
        preview_path = output_dir / "test_previews" / preview_name
        # preview shows top-3 summary
        _save_preview_image(
            source_image,
            gradcam_overlay,
            preview_path,
            top_text,
            "N/A",
            "; ".join(interpretation_lines),
            ", ".join([f"{p:.2f}" for _, p in topk]),
        )
        print(f"Saved preview image: {preview_path}")

    gradcam.close()


def _print_test_analysis_summary(stats: dict[str, float | int]) -> None:
    print("Confusion matrix:")
    print("                Pred No Finding   Pred Finding")
    print(f"Actual No Finding  {stats['tn']:>6}            {stats['fp']:>6}")
    print(f"Actual Finding     {stats['fn']:>6}            {stats['tp']:>6}")
    print(
        "Metrics | "
        f"Precision={stats['precision']:.4f} "
        f"Recall={stats['recall']:.4f} "
        f"F1={stats['f1_score']:.4f} "
        f"Accuracy={stats['accuracy']:.4f}"
    )


def save_checkpoint(model: nn.Module, class_names: Iterable[str], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "chest_xray_cnn.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "class_names": list(class_names),
    }
    torch.save(payload, checkpoint_path)
    metadata_path = output_dir / "chest_xray_classes.json"
    metadata_path.write_text(json.dumps({"class_names": list(class_names)}, indent=2))
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom chest X-ray CNN from scratch.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "Downloads" / "archive",
        help="Path to the chest X-ray dataset folder (default: ~/Downloads/archive).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50000,
        help="Maximum images to use for training (default: 50000). Set to 0 to use the full dataset.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Resize images to this square size.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory to save checkpoints.")
    parser.add_argument(
        "--external-test-folder",
        type=Path,
        default=Path("input"),
        help="Path to a folder of external images to run inference on after training (default: input/).",
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


def run_external_test_inference(
    model: nn.Module,
    images: list[tuple[Path, torch.Tensor]],
    class_names: list[str],
    device: torch.device,
    output_dir: Path,
    image_size: int,
):
    if not images:
        print("No external images found to run inference on.")
        return

    model.eval()
    gradcam = GradCAM(model, _get_last_conv_layer(model))
    output_dir = output_dir / "external_test_previews"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, (path, tensor) in enumerate(images, start=1):
        input_tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
            # top-3
            labelled = list(enumerate(probs))
            labelled_sorted = sorted(labelled, key=lambda x: x[1], reverse=True)
            pred = labelled_sorted[0][0]
            conf = labelled_sorted[0][1]

        # conservative wording
        topk_text = ", ".join([f"{class_names[i]} ({p:.2f})" for i, p in labelled_sorted[:3]])
        interpretation = []
        for i, p in labelled_sorted[:3]:
            if p >= 0.5:
                interpretation.append(f"Possible signs of {class_names[i]} detected ({p:.2f})")
            elif p >= 0.2:
                interpretation.append(f"Weak evidence of {class_names[i]} ({p:.2f})")
        if not interpretation:
            interpretation = ["No strong signs detected (low confidence)"]
        print(f"External {idx}: {path.name} | Top: {topk_text}")
        for line in interpretation:
            print("  -", line)

        source_image = Image.open(path).convert("RGB")
        heatmap = gradcam(input_tensor, pred)
        gradcam_overlay = _make_gradcam_overlay(source_image, heatmap)

        preview_name = f"external_preview_{idx}_{path.name}"
        preview_path = output_dir / preview_name
        _save_preview_image(
            source_image,
            gradcam_overlay,
            preview_path,
            class_names[pred],
            "N/A",
            "Prediction",
            f"{conf:.2f}",
        )
        print(f"Saved external preview: {preview_path}")
        row = {"filename": path.name, "preview": str(preview_path)}
        for i, name in enumerate(class_names):
            row[name] = probs[i]
        row["interpretation"] = " | ".join(interpretation)
        results.append(row)

    gradcam.close()
    # write CSV summary
    csv_path = output_dir / "external_test_results.csv"
    try:
        # write CSV with per-label probabilities
        fieldnames = ["filename"] + list(class_names) + ["preview", "interpretation"]
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Wrote external results CSV: {csv_path}")
    except Exception as exc:
        print(f"Failed to write external results CSV: {exc}")

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
    # Inference-only mode: load checkpoint and run external-folder inference
    if getattr(args, "inference_only", False):
        if args.external_test_folder is None:
            raise ValueError("--inference-only requires --external-test-folder to be set.")
        checkpoint_path = args.checkpoint
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=device)
        class_names = payload.get("class_names")
        if class_names is None:
            raise ValueError("Checkpoint missing class_names metadata.")
        model = ChestXRayCNN(num_classes=len(class_names)).to(device)
        model.load_state_dict(payload["model_state_dict"])
        external_images = _load_external_images(args.external_test_folder, args.image_size)
        run_external_test_inference(model, external_images, class_names, device, args.output_dir, args.image_size)
        return
    train_loader, val_loader, test_loader, class_names, pos_weight = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        max_images=args.max_images,
        image_size=args.image_size,
        seed=args.seed,
    )

    model = ChestXRayCNN(num_classes=len(class_names)).to(device)
    multi_label = len(class_names) > 2
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device)) if multi_label and pos_weight is not None else nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Using dataset root: {_resolve_image_root(args.data_root)}")
    print(f"Classes detected: {class_names}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    if args.max_images and args.max_images > 0:
        print(f"Training on at most {args.max_images} images")
    else:
        print("Training on full dataset")
    if multi_label and pos_weight is not None:
        print(f"Using class-imbalance weighting: pos_weight shape={tuple(pos_weight.shape)}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = __import__("time").time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        epoch_seconds = __import__("time").time() - epoch_start
        if multi_label:
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={epoch_seconds:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={epoch_seconds:.1f}s"
            )

    test_loss, test_targets, test_predictions, test_probabilities = collect_test_predictions(
        model,
        test_loader,
        criterion,
        device,
    )
    print(f"Test set | test_loss={test_loss:.4f}")

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
        }
        for i, name in enumerate(class_names):
            metrics["per_label"][name] = {
                "precision": float(per_label_precision[i]),
                "recall": float(per_label_recall[i]),
                "f1": float(per_label_f1[i]),
                "prevalence": int(targets_np[:, i].sum()),
            }
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

    preview_test_predictions(
        model,
        test_loader.dataset,
        class_names,
        device,
        args.output_dir,
        [],
        [],
        max_samples=5,
    )

    checkpoint_path = save_checkpoint(model, class_names, args.output_dir)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Run inference on the external input folder after training.
    if getattr(args, "external_test_folder", None) is not None:
        ext_folder = args.external_test_folder
        external_images = _load_external_images(ext_folder, args.image_size)
        run_external_test_inference(
            model,
            external_images,
            class_names,
            device,
            args.output_dir,
            args.image_size,
        )


if __name__ == "__main__":
    main()
