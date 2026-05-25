from __future__ import annotations

"""Load saved heart + skin checkpoints and evaluate on the same stratified test splits as training (seed=42)."""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from train_heart_tabular import (
    DEFAULT_DATA as DEFAULT_HEART_DATA,
    HeartMLP,
    load_and_split,
)
from train_skin_cnn import (
    DEFAULT_CSV as DEFAULT_SKIN_CSV,
    HmnistSubsetDataset,
    SkinLesionCNN,
    _build_eval_transform,
    load_hmnist_from_csv,
)

DEFAULT_HEART_CKPT = SCRIPT_DIR / "outputs" / "heart" / "heart_mlp.pt"
DEFAULT_HEART_PRE = SCRIPT_DIR / "outputs" / "heart" / "heart_preprocessor.joblib"
DEFAULT_SKIN_CKPT = SCRIPT_DIR / "outputs" / "skin" / "skin_cnn.pt"


def _torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
@torch.no_grad()
def _heart_test(
    data: Path,
    ckpt_path: Path,
    preprocessor_path: Path,
    device: torch.device,
    seed: int,
    sample_rows: int,
) -> None:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Heart checkpoint missing: {ckpt_path}")
    if not preprocessor_path.is_file():
        raise FileNotFoundError(f"Heart preprocessor missing: {preprocessor_path}")

    payload = _torch_load(ckpt_path, device)
    model = HeartMLP(
        input_dim=int(payload["input_dim"]),
        hidden1=int(payload["hidden1"]),
        hidden2=int(payload["hidden2"]),
        embed_dim=int(payload["embed_dim"]),
        num_classes=int(payload["num_classes"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    class_names: list[str] = list(payload["class_names"])

    pre = joblib.load(preprocessor_path)
    _, _, X_test_df, _, _, y_test = load_and_split(data, seed)
    X_test_np = pre.transform(X_test_df)
    xb = torch.tensor(X_test_np, dtype=torch.float32, device=device)
    logits = model(xb)
    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1).cpu().numpy()
    y_test = np.asarray(y_test)

    acc = accuracy_score(y_test, pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)

    print("\n=== Heart MLP (tabular) - test split ===")
    print(f"  Samples: {len(y_test)} | accuracy={acc:.4f} precision={p:.4f} recall={r:.4f} f1={f1:.4f}")
    if sample_rows > 0:
        n = min(sample_rows, len(y_test))
        print(f"  First {n} test rows (true -> pred, confidence):")
        for i in range(n):
            j = int(pred[i])
            conf = float(probs[i, j].item())
            print(
                f"    [{i}] true={class_names[int(y_test[i])]} -> pred={class_names[j]} "
                f"(p={conf:.3f})"
            )


@torch.no_grad()
def _skin_test(
    csv_path: Path,
    ckpt_path: Path,
    device: torch.device,
    seed: int,
    batch_size: int,
    sample_rows: int,
) -> None:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Skin checkpoint missing: {ckpt_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Skin CSV missing: {csv_path}")

    payload = _torch_load(ckpt_path, device)
    class_names: list[str] = list(payload["class_names"])
    image_size = int(payload["image_size"])
    num_classes = int(payload["num_classes"])

    model = SkinLesionCNN(num_classes=num_classes).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    images, labels = load_hmnist_from_csv(csv_path)
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=seed)
    _, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=seed)

    eval_t = _build_eval_transform(image_size)
    test_ds = HmnistSubsetDataset(images, labels, test_idx, eval_t)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_pred: list[int] = []
    all_true: list[int] = []

    for imgs, y in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        pred = logits.argmax(dim=1).cpu().numpy().tolist()
        all_pred.extend(pred)
        all_true.extend(y.numpy().tolist())

    acc = accuracy_score(all_true, all_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average="macro", zero_division=0
    )

    print("\n=== Skin CNN - test split ===")
    print(
        f"  Samples: {len(all_true)} | accuracy={acc:.4f} | "
        f"macro_p={macro_p:.4f} macro_r={macro_r:.4f} macro_f1={macro_f1:.4f}"
    )

    if sample_rows > 0:
        print(f"  First {min(sample_rows, len(all_true))} test samples (true -> pred):")
        for i in range(min(sample_rows, len(all_true))):
            print(
                f"    [{i}] true={class_names[int(all_true[i])]} -> pred={class_names[int(all_pred[i])]}"
            )

    out_metrics = {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "n_test": len(all_true),
    }
    out_path = SCRIPT_DIR / "outputs" / "skin" / "last_eval_test_run.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_metrics, indent=2))
    print(f"  Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved heart and skin models on test splits.")
    p.add_argument("--seed", type=int, default=42, help="Must match training seed for same splits.")
    p.add_argument("--heart-only", action="store_true")
    p.add_argument("--skin-only", action="store_true")
    p.add_argument("--heart-data", type=Path, default=DEFAULT_HEART_DATA)
    p.add_argument("--heart-checkpoint", type=Path, default=DEFAULT_HEART_CKPT)
    p.add_argument("--heart-preprocessor", type=Path, default=DEFAULT_HEART_PRE)
    p.add_argument("--skin-csv", type=Path, default=DEFAULT_SKIN_CSV)
    p.add_argument("--skin-checkpoint", type=Path, default=DEFAULT_SKIN_CKPT)
    p.add_argument("--skin-batch-size", type=int, default=64)
    p.add_argument("--sample-rows", type=int, default=5, help="Print this many example predictions per model (0 to skip).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_heart = not args.skin_only
    run_skin = not args.heart_only

    if run_heart:
        _heart_test(
            args.heart_data,
            args.heart_checkpoint,
            args.heart_preprocessor,
            device,
            args.seed,
            args.sample_rows,
        )
    if run_skin:
        _skin_test(
            args.skin_csv,
            args.skin_checkpoint,
            device,
            args.seed,
            args.skin_batch_size,
            args.sample_rows,
        )

    if run_heart and run_skin:
        print("\nBoth models evaluated. Check paths above if anything failed.")


if __name__ == "__main__":
    main()
