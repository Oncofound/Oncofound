"""cls_utils.py - CSV label discovery for one-hot classification labels."""
from __future__ import annotations
import pandas as pd


def discover_label_columns(df):
    candidate = [
        col for col in df.columns
        if set(df[col].dropna().unique()).issubset({0, 1})
    ]
    valid = [col for col in candidate if df[col].sum() > 0]
    one_hot_ok = df[valid].sum(axis=1).eq(1).all()
    if not one_hot_ok:
        raise ValueError("Auto-detected 0/1 columns are not a strict one-hot encoding.")
    return valid


def infer_num_classes_and_counts(csv_file):
    df = pd.read_csv(csv_file)
    label_cols = discover_label_columns(df)
    num_classes = len(label_cols)
    counts = {col: int(df[col].sum()) for col in label_cols}
    return num_classes, counts, label_cols
