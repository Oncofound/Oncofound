"""Validate the resolved release configuration without starting a job."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from dinov3.configs import DinoV3SetupArgs, get_cfg_from_args
from dinov3.data.loaders import _parse_dataset_str


def validate(stage, check_files=False, opts=None):
    cfg = get_cfg_from_args(DinoV3SetupArgs(
        config_file=str(ROOT / 'configs' / f'{stage}.yaml'),
        output_dir=f'./outputs/{stage}',
        opts=list(opts or []),
    ), strict=False)  # Match the real training entry: setup_config(strict_cfg=False).
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert cfg.student.arch == 'vit_large'
    assert cfg.student.in_chans == 1 and cfg.student.patch_size == 16
    assert cfg.crops.crop_setting == 2
    assert cfg.crops.global_crops_size == 96 and cfg.crops.local_crops_size == 48
    assert cfg.crops.local_crops_number == 8
    assert 0 <= cfg.train.unlabeled_dataset_ratio <= 1
    assert cfg.train.OFFICIAL_EPOCH_LENGTH > 0
    assert cfg.optim.epochs > 0
    assert cfg.train.batch_size_per_gpu > 0 and cfg.train.num_workers >= 0
    for ds in (cfg.train.dataset_path, cfg.train.unlabeled_dataset_path):
        _, kw = _parse_dataset_str(ds)
        assert kw['min_slice'] == 6
        if 'proposal_jsonl' in kw:
            assert kw['proposal_match_mode'] == 'legacy_basename'
        if check_files:
            for key in ('meta', 'proposal_jsonl'):
                if key in kw and not Path(kw[key]).is_file():
                    raise FileNotFoundError(f'{key}: {kw[key]}')
    if check_files and not Path(cfg.student.resume_from_teacher_chkpt).is_file():
        raise FileNotFoundError(cfg.student.resume_from_teacher_chkpt)
    return cfg


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['warmup', 'pretrain'], default='pretrain')
    p.add_argument('--check-files', action='store_true')
    p.add_argument('opts', nargs=argparse.REMAINDER)
    args = p.parse_args()
    validate(args.stage, args.check_files, args.opts)
    print(f'{args.stage}: config passed')
