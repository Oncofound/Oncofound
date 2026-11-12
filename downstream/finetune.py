"""Single-task classification finetuning over pre-extracted feature bags.

Trains the gated-attention classifier on all provided cases (no train/val/test
split) and saves a checkpoint. This entry intentionally does no evaluation and
computes no metrics; it only fits the classification head and writes weights.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .single_task.cls_dataset import FeatureDataset
from .single_task.cls_model import GateFusionClassifier
from .single_task.cls_utils import infer_num_classes_and_counts


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    num_classes, class_counts, label_cols = infer_num_classes_and_counts(args.csv)
    df = pd.read_csv(args.csv)
    df["PatientID"] = df["PatientID"].astype(str)

    # Infer feature dimension from the first available case.
    present = [pid for pid in df["PatientID"] if os.path.exists(os.path.join(args.feats_dir, f"{pid}.npy"))]
    if not present:
        raise RuntimeError("No feature files found for the cases listed in the CSV")
    first = np.load(os.path.join(args.feats_dir, f"{present[0]}.npy"))
    if first.ndim != 2:
        raise RuntimeError(f"Expected a 2D [N, D] feature bag, got {first.shape}")
    feature_dim = int(first.shape[1])
    print(f"Classes: {label_cols}  counts: {class_counts}  feature_dim: {feature_dim}")

    dataset = FeatureDataset(
        df=df,
        features_dir=args.feats_dir,
        label_cols=label_cols,
        sample=args.sample,
        patient_id_col="PatientID",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = GateFusionClassifier(
        emb_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        n_cls=num_classes,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for feats, labels, _ in loader:
            feats = feats.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            seen += 1
        avg = total / max(seen, 1)
        print(f"Epoch {epoch}/{args.epochs}  loss={avg:.4f}")

    checkpoint = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "feature_dim": feature_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "n_cls": num_classes,
        "label_cols": label_cols,
        "epoch": args.epochs,
    }
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(checkpoint, ckpt_path)
    meta = {
        "feature_dim": feature_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "n_cls": num_classes,
        "label_cols": label_cols,
        "epochs": args.epochs,
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved checkpoint to {ckpt_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="CSV with PatientID and one-hot label columns")
    p.add_argument("--feats_dir", required=True, help="Directory of <PatientID>.npy feature bags")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--sample", default="all", help="'all' or an integer to subsample patches")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
