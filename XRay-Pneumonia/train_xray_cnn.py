from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "xray"


class ChestPneumoniaCNN(nn.Module):
    """Four conv blocks (3x3, BN, ReLU) — same architecture as SkinLesionCNN."""

    def __init__(self, num_classes: int = 2) -> None:
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


def resolve_data_root(base: Path) -> Path:
    """Find Kaggle chest_xray folder (handles double-nested zip extract)."""
    candidates = [
        base / "chest_xray" / "chest_xray",
        base / "chest_xray",
        base,
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find train/ and test/ under {base}. "
        "Expected XRay-Pneumonia/chest_xray/chest_xray/{{train,test}}/."
    )


def _build_train_transform(image_size: int) -> transforms.Compose:
    pad = max(16, image_size // 8)
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size + pad, image_size + pad)),
            transforms.RandomResizedCrop(image_size, scale=(0.88, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.5, 2.0)),
        ]
    )


def _build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_dataloaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    seed: int,
    val_fraction: float = 0.15,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], list[int]]:
    train_root = data_root / "train"
    test_root = data_root / "test"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Missing train folder: {train_root}")
    if not test_root.is_dir():
        raise FileNotFoundError(f"Missing test folder: {test_root}")

    train_t = _build_train_transform(image_size)
    eval_t = _build_eval_transform(image_size)

    train_source = datasets.ImageFolder(str(train_root), transform=eval_t)
    class_names = list(train_source.classes)
    if len(class_names) < 2:
        raise ValueError(f"Expected at least 2 classes, got {class_names}")

    indices = np.arange(len(train_source))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        stratify=np.array(train_source.targets),
        random_state=seed,
    )

    train_ds = Subset(datasets.ImageFolder(str(train_root), transform=train_t), train_idx.tolist())
    val_ds = Subset(datasets.ImageFolder(str(train_root), transform=eval_t), val_idx.tolist())
    test_ds = datasets.ImageFolder(str(test_root), transform=eval_t)

    y_train = [train_source.targets[i] for i in train_idx]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, class_names, y_train


def _get_last_conv_layer(model: nn.Module) -> nn.Module:
    for layer in reversed(list(model.features)):
        if isinstance(layer, nn.Conv2d):
            return layer
    raise ValueError("No convolutional layer found for Grad-CAM.")


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

    def __call__(self, inputs: torch.Tensor, class_index: int) -> torch.Tensor:
        activations: torch.Tensor | None = None

        def forward_hook(_module, _inp, output: torch.Tensor) -> None:
            nonlocal activations
            activations = output
            output.retain_grad()

        handle = self.target_layer.register_forward_hook(forward_hook)
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(inputs)
        score = outputs[0, class_index]
        score.backward()

        handle.remove()
        if was_training:
            self.model.train()

        if activations is None or activations.grad is None:
            raise RuntimeError("Grad-CAM did not capture activation gradients.")

        weights = activations.grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)[0]
        # Min-max scale (same as skin CNN): ReLU-only fails when all weights*acts are negative,
        # which happens on high-confidence PNEUMONIA predictions before AdaptiveAvgPool.
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def close(self) -> None:
        return


def _tensor_to_pil_rgb(tensor_chw: torch.Tensor, image_size: int) -> Image.Image:
    x = tensor_chw.detach().cpu().clamp(-1, 1)
    x = (x * 0.5 + 0.5) * 255.0
    x = x.byte().numpy().transpose(1, 2, 0)
    return Image.fromarray(x, mode="RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)


def _chest_region_mask(image: Image.Image) -> np.ndarray:
    """Soft mask (H, W) in [0, 1] for the central thoracic/lung field."""
    gray = np.array(image.convert("L"), dtype=np.float32)
    h, w = gray.shape
    bg = float(np.percentile(gray, 5))
    body = gray > max(bg + 10.0, 22.0)

    # Use the middle band of the frame so edge tubes/leads do not define width.
    row_a, row_b = h // 5, 4 * h // 5
    col_occ = body[row_a:row_b].mean(axis=0)
    valid_cols = col_occ > 0.30
    if not valid_cols.any():
        return np.ones((h, w), dtype=np.float32)

    c0, c1 = int(np.flatnonzero(valid_cols)[0]), int(np.flatnonzero(valid_cols)[-1])
    body_rows = body[:, c0 : c1 + 1].any(axis=1)
    if not body_rows.any():
        return np.ones((h, w), dtype=np.float32)
    r0, r1 = int(np.flatnonzero(body_rows)[0]), int(np.flatnonzero(body_rows)[-1])

    inset_x = max(4, int((c1 - c0) * 0.14))
    inset_top = max(4, int((r1 - r0) * 0.14))
    inset_bot = max(4, int((r1 - r0) * 0.08))
    r0, r1 = r0 + inset_top, r1 - inset_bot
    c0, c1 = c0 + inset_x, c1 - inset_x
    if r0 >= r1 or c0 >= c1:
        return np.ones((h, w), dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    mask[r0 : r1 + 1, c0 : c1 + 1] = 1.0
    blurred = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2)),
        dtype=np.float32,
    )
    return blurred / 255.0


def _make_gradcam_overlay(image: Image.Image, heatmap: torch.Tensor, alpha: float = 0.45) -> Image.Image:
    base_size = image.size  # (width, height)
    heatmap_up = F.interpolate(
        heatmap.detach().unsqueeze(0).unsqueeze(0),
        size=(base_size[1], base_size[0]),
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()

    chest_mask = _chest_region_mask(image)
    heatmap_up = heatmap_up * chest_mask

    active = chest_mask > 0.05
    if active.any():
        vals = heatmap_up[active]
        if vals.max() > 0:
            cutoff = float(np.percentile(vals, 30))
            heatmap_up = np.where(heatmap_up >= cutoff, heatmap_up, 0.0)
            peak = float(heatmap_up[active].max())
            if peak > 0:
                heatmap_up = heatmap_up / peak

    base = np.array(image.convert("RGB"), dtype=np.float32)
    heatmap_colored = colormaps["jet"](heatmap_up)[..., :3] * 255.0
    mask_rgb = chest_mask[..., np.newaxis]
    blended = base * (1.0 - alpha * mask_rgb) + heatmap_colored * (alpha * mask_rgb)
    return Image.fromarray(blended.astype(np.uint8))


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


def _balanced_sample_indices(dataset: Dataset, class_names: list[str], max_samples: int) -> list[int]:
    """Pick examples from each class (ImageFolder stores NORMAL first, then PNEUMONIA)."""
    if not hasattr(dataset, "targets"):
        return list(range(min(max_samples, len(dataset))))

    by_label: dict[int, list[int]] = {i: [] for i in range(len(class_names))}
    for index, label in enumerate(dataset.targets):
        if label in by_label:
            by_label[label].append(index)

    per_class = max(1, max_samples // max(1, len(class_names)))
    selected: list[int] = []
    for label in range(len(class_names)):
        selected.extend(by_label.get(label, [])[:per_class])
    if len(selected) < max_samples:
        for label in range(len(class_names)):
            for index in by_label.get(label, [])[per_class:]:
                if index not in selected:
                    selected.append(index)
                if len(selected) >= max_samples:
                    break
            if len(selected) >= max_samples:
                break
    return selected[:max_samples]


def preview_gradcam_samples(
    model: nn.Module,
    dataset: Dataset,
    class_names: list[str],
    device: torch.device,
    output_dir: Path,
    image_size: int,
    max_samples: int = 5,
    preview_subdir: str = "test_previews",
) -> None:
    sample_indices = _balanced_sample_indices(dataset, class_names, max_samples)
    if not sample_indices:
        return
    gradcam = GradCAM(model, _get_last_conv_layer(model))
    out = output_dir / preview_subdir
    print(f"Saving {len(sample_indices)} Grad-CAM previews to {out}", flush=True)
    for preview_num, i in enumerate(sample_indices, start=1):
        img_tensor, label = dataset[i]
        input_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred = int(probs.argmax().item())
            conf = float(probs[pred].item())
        with torch.enable_grad():
            heatmap = gradcam(input_tensor, pred)
        src = _tensor_to_pil_rgb(img_tensor, image_size)
        overlay = _make_gradcam_overlay(src, heatmap)
        name = f"preview_{preview_num}_pred_{class_names[pred]}_true_{class_names[label]}.png"
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


def _save_checkpoint(
    path: Path,
    class_names: list[str],
    image_size: int,
    num_classes: int,
    state_dict: dict[str, torch.Tensor],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state_dict,
            "class_names": class_names,
            "image_size": image_size,
            "num_classes": num_classes,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train chest X-ray pneumonia CNN (NORMAL vs PNEUMONIA).")
    p.add_argument(
        "--data-root",
        type=Path,
        default=SCRIPT_DIR,
        help="Base folder containing chest_xray/ (default: XRay-Pneumonia/).",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradcam-samples", type=int, default=5, help="Test samples for Grad-CAM PNGs (0 to skip).")
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

    data_root = resolve_data_root(args.data_root)
    print(f"Using dataset root: {data_root}", flush=True)

    train_loader, val_loader, test_loader, class_names, y_train = build_dataloaders(
        data_root=data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    num_classes = len(class_names)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=np.array(y_train),
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    model = ChestPneumoniaCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    print(
        f"Device: {device} | classes={class_names} | "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}",
        flush=True,
    )

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    best_path = args.output_dir / "xray_cnn_best.pt"

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model, train_loader, criterion, optimizer, device, log_interval=max(20, len(train_loader) // 5)
        )
        va = evaluate_loss(model, val_loader, criterion, device)
        scheduler.step(va)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={tr:.4f} | val_loss={va:.4f} | lr={lr:.6f}",
            flush=True,
        )
        if va < best_val:
            best_val = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            _save_checkpoint(best_path, class_names, args.image_size, num_classes, best_state)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(val_loss did not improve for {args.early_stopping_patience} epochs).",
                    flush=True,
                )
                break

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
        "data_root": str(data_root),
    }
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        tgt_test, pred_test, labels=list(range(num_classes)), average=None, zero_division=0
    )
    metrics["per_class"] = {
        class_names[i]: {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f1[i]),
        }
        for i in range(num_classes)
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"Test | loss={test_loss:.4f} | acc={acc:.4f} | macro_f1={macro_f1:.4f} | micro_f1={micro_f1:.4f}",
        flush=True,
    )

    ckpt_path = args.output_dir / "xray_cnn.pt"
    final_state = best_state if best_state is not None else {k: v.cpu().clone() for k, v in model.state_dict().items()}
    _save_checkpoint(ckpt_path, class_names, args.image_size, num_classes, final_state)
    print(f"Saved checkpoint: {ckpt_path}", flush=True)
    print(f"Best val checkpoint: {best_path}", flush=True)

    if args.gradcam_samples > 0:
        preview_gradcam_samples(
            model,
            test_loader.dataset,
            class_names,
            device,
            args.output_dir,
            args.image_size,
            args.gradcam_samples,
        )


if __name__ == "__main__":
    main()
