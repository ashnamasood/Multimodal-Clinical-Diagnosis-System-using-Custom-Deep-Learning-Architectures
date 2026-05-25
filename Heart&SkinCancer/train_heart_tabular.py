from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "Heart Data set" / "heart.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "heart"

CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
NUMERIC_COLS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
TARGET_COL = "HeartDisease"


class HeartMLP(nn.Module):
    """3 hidden layers ending in a 32-d embedding, then binary logits (for late fusion)."""

    def __init__(
        self,
        input_dim: int,
        hidden1: int = 64,
        hidden2: int = 64,
        embed_dim: int = 32,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        return self.head(z)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
            ("num", RobustScaler(), NUMERIC_COLS),
        ],
        remainder="drop",
    )


def load_and_split(
    csv_path: Path,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    for col in CATEGORICAL_COLS + NUMERIC_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {csv_path}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target {TARGET_COL}")

    X = df[CATEGORICAL_COLS + NUMERIC_COLS].copy()
    y = df[TARGET_COL].to_numpy()

    indices = np.arange(len(df))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices,
        y,
        test_size=0.3,
        stratify=y,
        random_state=seed,
    )
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=seed,
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)

    return X_train, X_val, X_test, y[train_idx], y[val_idx], y[test_idx]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
        n += xb.size(0)
    return running / max(1, n)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    running = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        running += loss.item() * xb.size(0)
        n += xb.size(0)
    return running / max(1, n)


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    preds: list[int] = []
    targets: list[int] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu().tolist()
        preds.extend(pred)
        targets.extend(yb.tolist())
    return preds, targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train tabular MLP for heart failure prediction (RobustScaler + 32-d embed).")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to heart.csv")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--hidden1", type=int, default=64)
    p.add_argument("--hidden2", type=int, default=64)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = load_and_split(args.data, args.seed)

    pre = build_preprocessor()
    X_train_np = pre.fit_transform(X_train_df)
    X_val_np = pre.transform(X_val_df)
    X_test_np = pre.transform(X_test_df)

    input_dim = X_train_np.shape[1]
    joblib.dump(
        pre,
        args.output_dir / "heart_preprocessor.joblib",
    )

    class_names = ["No heart disease", "Heart disease"]

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_np, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_np, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test_np, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = HeartMLP(
        input_dim=input_dim,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        embed_dim=args.embed_dim,
        num_classes=2,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Device: {device} | input_dim={input_dim} | train={len(y_train)} val={len(y_val)} test={len(y_test)}")

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss = evaluate_loss(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = evaluate_loss(model, test_loader, criterion, device)
    pred_test, tgt_test = collect_predictions(model, test_loader, device)
    acc = accuracy_score(tgt_test, pred_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        tgt_test, pred_test, average="binary", zero_division=0
    )

    metrics = {
        "test_loss": float(test_loss),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "best_val_loss": float(best_val),
        "input_dim": int(input_dim),
        "embed_dim": int(args.embed_dim),
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"Test | loss={test_loss:.4f} | accuracy={acc:.4f} | precision={precision:.4f} | "
        f"recall={recall:.4f} | f1={f1:.4f}"
    )

    ckpt_path = args.output_dir / "heart_mlp.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "input_dim": input_dim,
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
            "embed_dim": args.embed_dim,
            "num_classes": 2,
            "preprocessor_path": "heart_preprocessor.joblib",
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved preprocessor: {args.output_dir / 'heart_preprocessor.joblib'}")


if __name__ == "__main__":
    main()
