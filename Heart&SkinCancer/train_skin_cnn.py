from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "Skin Cancer Data set" / "hmnist_28_28_RGB.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "skin"

# HAM10000 MNIST (Kaggle) label order: indices 0..6
HMNIST_CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"]


class SkinLesionCNN(nn.Module):
    """Four conv blocks (3x3, BN, ReLU) matching the chest-X-ray custom CNN style."""

    def __init__(self, num_classes: int = 7) -> None:
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


def _build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def load_hmnist_from_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load HMNIST RGB rows once: images (N,H,W,3) uint8, labels (N,) int64."""
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"Unexpected CSV shape {df.shape} in {csv_path}")
    pixels = df.iloc[:, :-1].to_numpy(dtype=np.uint8)
    labels = df.iloc[:, -1].to_numpy(dtype=np.int64)
    side = int(round((pixels.shape[1] // 3) ** 0.5))
    if side * side * 3 != pixels.shape[1]:
        raise ValueError(f"Expected flat RGB size H*W*3, got n_cols={pixels.shape[1]} from {csv_path}")
    images = pixels.reshape(-1, side, side, 3)
    return images, labels


class HmnistSubsetDataset(Dataset):
    """View into shared in-memory HMNIST arrays (avoids reloading CSV per split)."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray, transform) -> None:
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = int(self.indices[index])
        img = Image.fromarray(self.images[row], mode="RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[row])


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


def _tensor_to_pil_rgb(tensor_chw: torch.Tensor, image_size: int) -> Image.Image:
    x = tensor_chw.detach().cpu().clamp(-1, 1)
    x = (x * 0.5 + 0.5) * 255.0
    x = x.byte().numpy().transpose(1, 2, 0)
    return Image.fromarray(x, mode="RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)


def _make_gradcam_overlay(image: Image.Image, heatmap: torch.Tensor, alpha: float = 0.4) -> Image.Image:
    heatmap_np = heatmap.detach().cpu().numpy()
    heatmap_colored = colormaps["jet"](heatmap_np)[..., :3]
    heatmap_image = Image.fromarray((heatmap_colored * 255).astype(np.uint8)).resize(image.size)
    base_image = image.convert("RGB")
    return Image.blend(base_image, heatmap_image, alpha=alpha)


def _save_preview_image(
    image: Image.Image,
    gradcam_overlay: Image.Image,
    output_path: Path,
    prediction_text: str,
    true_text: str,
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
    ]
    y = max(preview.height, gradcam_preview.height) + 8
    for line in lines:
        draw.text((8, y), line, fill=(0, 0, 0), font=font)
        y += 20
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def preview_gradcam_samples(
    model: nn.Module,
    dataset: Dataset,
    class_names: list[str],
    device: torch.device,
    output_dir: Path,
    image_size: int,
    max_samples: int = 5,
) -> None:
    n = min(max_samples, len(dataset))
    if n == 0:
        return
    gradcam = GradCAM(model, _get_last_conv_layer(model))
    out = output_dir / "val_previews"
    print(f"Saving {n} Grad-CAM previews to {out}", flush=True)
    for i in range(n):
        img_tensor, label = dataset[i]
        input_tensor = img_tensor.unsqueeze(0).to(device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred = int(probs.argmax().item())
        conf = float(probs[pred].item())
        src = _tensor_to_pil_rgb(img_tensor, image_size)
        heatmap = gradcam(input_tensor, pred)
        overlay = _make_gradcam_overlay(src, heatmap)
        name = f"preview_{i + 1}_pred_{class_names[pred]}_true_{class_names[label]}.png"
        _save_preview_image(
            src,
            overlay,
            out / name,
            class_names[pred],
            class_names[label],
            f"{conf:.2f}",
        )
    gradcam.close()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int = 50,
) -> float:
    model.train()
    running = 0.0
    total = 0
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
        total += labels.size(0)
        if log_interval > 0 and batch_index % log_interval == 0:
            print(
                f"    batch {batch_index}/{len(loader)} | avg_loss={running / max(1, total):.4f}",
                flush=True,
            )
    return running / max(1, total)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    running = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running += loss.item() * images.size(0)
        total += labels.size(0)
    return running / max(1, total)


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], float]:
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    loss_sum = 0.0
    n = 0
    crit = nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss_sum += crit(logits, labels).item() * images.size(0)
        n += labels.size(0)
        pred = logits.argmax(dim=1)
        all_preds.extend(pred.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())
    return all_preds, all_targets, loss_sum / max(1, n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train custom skin CNN (HAM10000 MNIST CSV or ImageFolder).")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to hmnist_*_RGB.csv (ignored if --image-root set).")
    p.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="If set, train with torchvision ImageFolder on this directory.",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradcam-samples", type=int, default=5, help="Val samples for Grad-CAM PNGs (0 to skip).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_t = _build_train_transform(args.image_size)
    eval_t = _build_eval_transform(args.image_size)

    if args.image_root is not None:
        root = args.image_root
        if not root.is_dir():
            raise FileNotFoundError(f"--image-root is not a directory: {root}")
        full = datasets.ImageFolder(str(root), transform=eval_t)
        class_names = list(full.classes)
        num_classes = len(class_names)
        if num_classes < 2:
            raise ValueError("ImageFolder needs at least 2 classes.")
        generator = torch.Generator().manual_seed(args.seed)
        n_total = len(full)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        n_test = n_total - n_train - n_val
        train_subset, val_subset, test_subset = random_split(full, [n_train, n_val, n_test], generator=generator)

        train_ds = Subset(datasets.ImageFolder(str(root), transform=train_t), train_subset.indices)
        val_ds = Subset(datasets.ImageFolder(str(root), transform=eval_t), val_subset.indices)
        test_ds = Subset(datasets.ImageFolder(str(root), transform=eval_t), test_subset.indices)
        y_train = [full.targets[i] for i in train_subset.indices]
    else:
        if not args.csv.exists():
            raise FileNotFoundError(f"CSV not found: {args.csv}. Download hmnist_28_28_RGB.csv or pass --image-root.")
        print(f"Loading HMNIST CSV once: {args.csv}", flush=True)
        images, labels = load_hmnist_from_csv(args.csv)
        class_names = HMNIST_CLASS_NAMES
        num_classes = len(class_names)
        indices = np.arange(len(labels))
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=args.seed)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=args.seed
        )
        train_ds = HmnistSubsetDataset(images, labels, train_idx, train_t)
        val_ds = HmnistSubsetDataset(images, labels, val_idx, eval_t)
        test_ds = HmnistSubsetDataset(images, labels, test_idx, eval_t)
        y_train = labels[train_idx].tolist()

    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=np.array(y_train))
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SkinLesionCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Device: {device} | classes={num_classes} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, log_interval=max(50, len(train_loader)))
        va = evaluate_loss(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={tr:.4f} | val_loss={va:.4f}", flush=True)
        if va < best_val:
            best_val = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_test, tgt_test, test_loss = evaluate_metrics(model, test_loader, device)
    acc = accuracy_score(tgt_test, pred_test)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        tgt_test, pred_test, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        tgt_test, pred_test, average="micro", zero_division=0
    )

    metrics = {
        "test_loss": float(test_loss),
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "best_val_loss": float(best_val),
        "class_names": class_names,
    }
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        tgt_test, pred_test, labels=list(range(num_classes)), average=None, zero_division=0
    )
    metrics["per_class"] = {
        class_names[i]: {"precision": float(per_class_p[i]), "recall": float(per_class_r[i]), "f1": float(per_class_f1[i])}
        for i in range(num_classes)
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"Test | loss={test_loss:.4f} | acc={acc:.4f} | macro_f1={macro_f1:.4f} | micro_f1={micro_f1:.4f}",
        flush=True,
    )

    ckpt_path = args.output_dir / "skin_cnn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": args.image_size,
            "num_classes": num_classes,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}", flush=True)

    if args.gradcam_samples > 0:
        preview_gradcam_samples(model, val_ds, class_names, device, args.output_dir, args.image_size, args.gradcam_samples)


if __name__ == "__main__":
    main()
