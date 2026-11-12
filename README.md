# OncoFound — Pancancer 3D DINO

Self-supervised 3D foundation-model training, feature extraction, and downstream
classification finetuning for multi-modal pancancer medical imaging.

The repository contains three parts:

1. **Pretraining** — 3D DINOv3 (ViT-Large, patch 16) with iBOT, warmup + pretrain.
2. **Feature extraction** — local crop sampler (CoarseLocNet) + DINO encoder,
   producing per-case feature bags (`.npy`, shape `[N_patches, 1024]`).
3. **Downstream finetuning** — gated-attention classifier over cached feature bags.

No train/validation/test split logic and no metric computation are included.

## Installation

Python 3.11 and a CUDA-capable PyTorch installation are recommended.

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

## 1. Pretraining

Set data and checkpoint paths in `configs/warmup.yaml` and `configs/pretrain.yaml`,
then run:

```bash
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0 bash scripts/launch.sh warmup
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0 bash scripts/launch.sh pretrain
```

## 2. Feature extraction

Generate local-crop proposals and extract features:

```bash
python scripts/select_local_crops.py --ckpt ./weights/sampler/best_all8_hit.pt \
    --manifest ./data/cases.csv --output-dir outputs/proposals
python scripts/extract_features.py --proposals ./data/proposals.jsonl \
    --checkpoint ./weights/dinov3/teacher_checkpoint.pth \
    --output-dir ./data/features/example_cohort
```

## 3. Downstream finetuning

Train the classification head over pre-extracted feature bags:

```bash
python -m downstream.finetune \
    --csv ./data/labels.csv \
    --feats_dir ./data/features/example_cohort \
    --output_dir outputs/classification \
    --hidden_dim 512 --dropout 0.3 --epochs 20 --lr 1e-4
```

The CSV lists one row per case with a `PatientID` column and one-hot class columns
(e.g. `Class0, Class1`).

## Inference demo notebook

`notebooks/inference_demo.ipynb`:

1. Builds the pretraining backbone and shows its output dimensions on dummy data.
2. Loads a released feature bag + downstream checkpoint and runs forward to the
   classification head, printing the logits.

## Weights and demo data

- Downstream classification checkpoint: download from the **GitHub Release** of this
  repository and place it under `weights/classification_best.pt`.
- Pretraining weights are not released; use your own checkpoints for training.
- A 20-case demo feature set (MRI I-SPY1 laterality classification, 10 left + 10
  right) is provided under `data/demo/` (features in `data/demo/features/`, labels in
  `data/demo/labels.csv`).

## License

See [LICENSE.md](LICENSE.md). The `dinov3` package is derived from Meta's DINOv3 and
retains its license header.
