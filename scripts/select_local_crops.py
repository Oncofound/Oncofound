"""Select unlabeled local crops using the published CoarseLocNet configuration.

Outputs one selected box per JSONL line, directly accepted by the proposal
converter. No segmentation or evaluation labels are required.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nibabel as nib
import torch
from lesion_prior.config import load_config
from lesion_prior.checkpoint import build_model_from_checkpoint
from lesion_prior.dataset import LesionPriorDataset
from lesion_prior.infer_sw import sliding_window_predict
from lesion_prior.crop_select import select_topk_local_crops, select_topk_local_crops_gaussian


def select_boxes(prob, shape, cfg, selector):
    common = dict(prob_map=prob, volume_shape=tuple(shape),
                  token_stride=tuple(cfg.target.token_stride),
                  crop_size=tuple(cfg.crop_eval.local_crop_size),
                  k=cfg.crop_eval.num_local_crops)
    if selector == 'gaussian':
        return select_topk_local_crops_gaussian(**common)
    return select_topk_local_crops(
        **common, nms_iou_threshold=cfg.crop_eval.nms_iou_threshold,
        min_center_distance=tuple(cfg.crop_eval.min_center_distance),
        threshold=cfg.crop_eval.threshold, fallback_to_grid=cfg.crop_eval.fallback_to_grid)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'configs/crop_light_train.yaml'))
    p.add_argument('--ckpt', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--split', default='all')
    p.add_argument('--selector', choices=['both', 'gaussian', 'nms'], default='both')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--roi-size', type=int, nargs=3)
    p.add_argument('--sw-batch-size', type=int)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args(argv)
    torch.set_num_threads(max(1, args.threads))
    cfg = load_config(args.config)
    if tuple(cfg.target.token_stride) != (8, 8, 8):
        raise ValueError('CoarseLocNet has fixed token stride 8 on every axis')
    if not cfg.data.orientation_ras or not cfg.data.full_volume_eval:
        raise ValueError('Proposal conversion requires RAS orientation and full-volume inference')
    if args.roi_size:
        cfg.data.sw_roi_size = tuple(args.roi_size)
    if any(n < 8 or n % 8 for n in cfg.data.sw_roi_size):
        raise ValueError('Inference ROI must be positive multiples of 8')
    if args.sw_batch_size:
        cfg.data.sw_batch_size = args.sw_batch_size
    if cfg.data.sw_batch_size < 1:
        raise ValueError('Sliding-window batch size must be positive')
    with open(args.manifest, newline='') as f:
        rows = list(csv.DictReader(f))
    if args.split != 'all':
        rows = [r for r in rows if r.get(cfg.data.split_key) == args.split]
    if not rows:
        raise ValueError('Manifest contains no selected cases')
    identifiers = [r.get(cfg.data.case_id_key, '') for r in rows]
    if any(not c for c in identifiers) or len(set(identifiers)) != len(rows):
        raise ValueError('Each selected row must have a nonempty unique case_id')
    # This bridge targets the single 3D image per row consumed by DINO.
    # Other layouts remain available through lesion_prior.test.
    for row in rows:
        image = row[cfg.data.image_key]
        if '|' in image or len(nib.load(image).shape) != 3:
            raise ValueError('The DINO proposal bridge requires one 3D NIfTI image per row')
        row[cfg.data.split_key] = 'inference'
    dataset = LesionPriorDataset(cfg.data, cfg.target, split='inference',
                                 manifest_rows=rows, augment=False, compute_targets=False)
    device = torch.device(args.device)
    model = build_model_from_checkpoint(args.ckpt, cfg).to(device).eval()
    selectors = ['gaussian', 'nms'] if args.selector == 'both' else [args.selector]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as td:
        counts = {s: 0 for s in selectors}
        handles = {s: open(Path(td)/f'selected_{s}.jsonl', 'w') for s in selectors}
        try:
            for i, row in enumerate(rows):
                sample = dataset[i]
                if sample['case_id'] != row[cfg.data.case_id_key]:
                    raise RuntimeError('Input failed to load; dataset attempted to substitute another case')
                prob = sliding_window_predict(
                    model, sample['image'], sample['mod_mask'], cfg, device,
                    amp=device.type == 'cuda', sw_batch_size=cfg.data.sw_batch_size,
                    blend_mode=cfg.data.sw_blend_mode)
                if not torch.isfinite(prob).all():
                    raise RuntimeError('Inference produced non-finite probabilities')
                for selector in selectors:
                    boxes = select_boxes(prob, sample['image'].shape[1:], cfg, selector)
                    if not boxes:
                        raise RuntimeError(f'{selector} found no crop candidates')
                    for box in boxes:
                        handles[selector].write(json.dumps(
                            {'case_id': sample['case_id'], **box.to_dict()})+'\n')
                    counts[selector] += len(boxes)
        finally:
            for handle in handles.values():
                handle.close()
        for selector in selectors:
            os.replace(Path(td)/f'selected_{selector}.jsonl', out/f'selected_{selector}.jsonl')
    report = {'cases': len(rows), 'boxes': counts, 'checkpoint': Path(args.ckpt).name,
              'spacing': list(cfg.data.spacing), 'token_stride': list(cfg.target.token_stride),
              'crop_size': list(cfg.crop_eval.local_crop_size), 'roi_size': list(cfg.data.sw_roi_size)}
    (out/'selection_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
