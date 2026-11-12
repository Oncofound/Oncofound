"""cls_dataset.py - load pre-extracted 2-global + 8-local feature arrays."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, df, features_dir, label_cols, transform=None, sample="all", patient_id_col="PatientID"):
        self.df = df.reset_index(drop=True).copy()
        self.features_dir = features_dir
        self.label_cols = label_cols if isinstance(label_cols, list) else [label_cols]
        self.transform = transform
        self.sample = sample
        self.patient_id_col = patient_id_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row[self.patient_id_col])
        feat_path = os.path.join(self.features_dir, f"{pid}.npy")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f"Feature file not found for PatientID={pid}: {feat_path}"
            )
        feats = np.load(feat_path, allow_pickle=False)
        if isinstance(self.sample, list):
            n_sample = np.random.choice(self.sample)
        elif self.sample != "all":
            n_sample = int(self.sample)
        else:
            n_sample = None
        if n_sample is not None and feats.shape[0] > n_sample:
            sel = np.random.choice(feats.shape[0], n_sample, replace=False)
            feats = feats[sel]
        if self.transform:
            feats = self.transform(feats)
        feats = torch.tensor(feats, dtype=torch.float)
        labels = row[self.label_cols].values.astype(int)
        if len(self.label_cols) == 1:
            class_idx = int(labels[0])
        else:
            class_idx = int(np.argmax(labels))
        return feats, class_idx, pid
