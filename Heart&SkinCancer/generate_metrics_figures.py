from __future__ import annotations

"""
Generate PNG figures for heart + skin: metric bars, per-class F1, confusion matrices.
Reads outputs/*/metrics.json and re-runs test-split forward pass for confusion matrices.

Usage (from repo root):
  python "Heart&SkinCancer/generate_metrics_figures.py"
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import joblib  # noqa: E402

from train_heart_tabular import DEFAULT_DATA as DEFAULT_HEART_DATA  # noqa: E402
from train_heart_tabular import HeartMLP, load_and_split  # noqa: E402
from train_skin_cnn import (  # noqa: E402
    DEFAULT_CSV as DEFAULT_SKIN_CSV,
    HmnistSubsetDataset,
    SkinLesionCNN,
    _build_eval_transform,
    load_hmnist_from_csv,
)

FIG_DIR = SCRIPT_DIR / "outputs" / "figures"
HEART_METRICS = SCRIPT_DIR / "outputs" / "heart" / "metrics.json"
SKIN_METRICS = SCRIPT_DIR / "outputs" / "skin" / "metrics.json"
DEFAULT_SEED = 42


def _torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _get_heart_preds(
    data: Path,
    ckpt: Path,
    pre: Path,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    payload = _torch_load(ckpt, device)
    model = HeartMLP(
        input_dim=int(payload["input_dim"]),
        hidden1=int(payload["hidden1"]),
        hidden2=int(payload["hidden2"]),
        embed_dim=int(payload["embed_dim"]),
        num_classes=int(payload["num_classes"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    names = list(payload["class_names"])
    preprocessor = joblib.load(pre)
    _, _, X_test, _, _, y_test = load_and_split(data, seed)
    x = preprocessor.transform(X_test)
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device))
        pred = logits.argmax(dim=1).cpu().numpy()
    return np.asarray(y_test), pred, names


def _get_skin_preds(
    csv_path: Path,
    ckpt: Path,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    payload = _torch_load(ckpt, device)
    names = list(payload["class_names"])
    image_size = int(payload["image_size"])
    model = SkinLesionCNN(num_classes=int(payload["num_classes"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    images, labels = load_hmnist_from_csv(csv_path)
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=seed)
    _, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=seed)
    loader = DataLoader(
        HmnistSubsetDataset(images, labels, test_idx, _build_eval_transform(image_size)),
        batch_size=batch_size,
        shuffle=False,
    )
    all_p: list[int] = []
    all_t: list[int] = []
    with torch.no_grad():
        for imgs, y in loader:
            imgs = imgs.to(device)
            pred = model(imgs).argmax(dim=1).cpu().numpy().tolist()
            all_p.extend(pred)
            all_t.extend(y.numpy().tolist())
    return np.array(all_t), np.array(all_p), names


def _bar_metrics(path: Path, title: str, labels: list[str], values: list[float], y_lim: tuple[float, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.25, 0.75, len(labels)))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(y_lim if y_lim else (0, max(1.0, max(values) * 1.15)))
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _confusion_matrix_fig(path: Path, y_true: np.ndarray, y_pred: np.ndarray, display_labels: list[str], title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(display_labels))))
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _per_class_bars(path: Path, class_names: list[str], scores: dict[str, dict[str, float]], metric: str, title: str) -> None:
    vals = [scores[c][metric] for c in class_names]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(class_names))
    ax.bar(x, vals, color=plt.cm.Set3(np.linspace(0, 1, len(class_names))))
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_ylim(0, 1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Heart -----
    heart_ckpt = SCRIPT_DIR / "outputs" / "heart" / "heart_mlp.pt"
    heart_pre = SCRIPT_DIR / "outputs" / "heart" / "heart_preprocessor.joblib"
    if heart_ckpt.is_file() and heart_pre.is_file() and HEART_METRICS.is_file():
        meta = json.loads(HEART_METRICS.read_text(encoding="utf-8"))
        _bar_metrics(
            FIG_DIR / "01_heart_test_metrics_bar.png",
            "Heart MLP — test split (held-out 15%)",
            ["Accuracy", "Precision", "Recall", "F1"],
            [meta["accuracy"], meta["precision"], meta["recall"], meta["f1"]],
            y_lim=(0, 1.05),
        )
        y_t, y_p, h_names = _get_heart_preds(DEFAULT_HEART_DATA, heart_ckpt, heart_pre, device, DEFAULT_SEED)
        _confusion_matrix_fig(
            FIG_DIR / "02_heart_confusion_matrix.png",
            y_t,
            y_p,
            h_names,
            f"Heart — confusion (n={len(y_t)})\nCode: test_models.py + train_heart_tabular.py",
        )

    # ----- Skin -----
    skin_ckpt = SCRIPT_DIR / "outputs" / "skin" / "skin_cnn.pt"
    if skin_ckpt.is_file() and SKIN_METRICS.is_file():
        meta = json.loads(SKIN_METRICS.read_text(encoding="utf-8"))
        _bar_metrics(
            FIG_DIR / "03_skin_overall_metrics_bar.png",
            "Skin CNN — test split: overall scores",
            ["Accuracy", "Macro P", "Macro R", "Macro F1", "Micro F1"],
            [
                meta["accuracy"],
                meta["macro_precision"],
                meta["macro_recall"],
                meta["macro_f1"],
                meta["micro_f1"],
            ],
            y_lim=(0, 1.05),
        )
        per = meta["per_class"]
        classes = list(meta["class_names"])
        _per_class_bars(
            FIG_DIR / "04_skin_per_class_f1.png",
            classes,
            per,
            "f1",
            "Skin CNN — per-class F1 (test)",
        )
        _per_class_bars(
            FIG_DIR / "05_skin_per_class_precision.png",
            classes,
            per,
            "precision",
            "Skin CNN — per-class precision (test)",
        )
        _per_class_bars(
            FIG_DIR / "06_skin_per_class_recall.png",
            classes,
            per,
            "recall",
            "Skin CNN — per-class recall (test)",
        )
        y_t, y_p, s_names = _get_skin_preds(DEFAULT_SKIN_CSV, skin_ckpt, device, DEFAULT_SEED, 64)
        _confusion_matrix_fig(
            FIG_DIR / "07_skin_confusion_matrix.png",
            y_t,
            y_p,
            s_names,
            f"Skin — confusion (n={len(y_t)})\nCode: test_models.py + train_skin_cnn.py",
        )

    # ----- Combined dashboard-style summary -----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if HEART_METRICS.is_file():
        hm = json.loads(HEART_METRICS.read_text(encoding="utf-8"))
        axes[0].bar(
            ["Acc", "P", "R", "F1"],
            [hm["accuracy"], hm["precision"], hm["recall"], hm["f1"]],
            color=["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"],
        )
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title("Heart MLP (test)")
    if SKIN_METRICS.is_file():
        sm = json.loads(SKIN_METRICS.read_text(encoding="utf-8"))
        axes[1].bar(
            ["Acc", "macroF1", "microF1"],
            [sm["accuracy"], sm["macro_f1"], sm["micro_f1"]],
            color=["#1abc9c", "#34495e", "#f39c12"],
        )
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Skin CNN (test)")
    fig.suptitle("FusionNet-Scratch — model comparison (held-out test)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "00_summary_heart_vs_skin.png", dpi=150)
    plt.close(fig)

    print(f"Wrote figures under: {FIG_DIR}")


if __name__ == "__main__":
    main()
