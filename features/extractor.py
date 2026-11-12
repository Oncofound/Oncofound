"""Strict 3D DINO loading and portable [2 global + 8 local, 1024] features."""
from pathlib import Path
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from dinov3.models import build_model_from_cfg
from .legacy_proposal import _centered_crop, _random_global_center


def load_backbone(checkpoint, config, role='teacher', device='cpu'):
    if role not in ('teacher', 'student'):
        raise ValueError('role must be teacher or student')
    cfg = OmegaConf.load(config)
    model, dim = build_model_from_cfg(cfg, only_teacher=True)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if role in state:
        state = state[role]
    elif any(k in state for k in ('teacher','student')):
        raise ValueError(f'Checkpoint does not contain requested {role} weights')
    if not isinstance(state, dict) or not state:
        raise ValueError('Expected a tensor state_dict')
    if all(k.startswith('module.') for k in state):
        state = {k[7:]:v for k,v in state.items()}
    backbone = {k[len('backbone.'):]:v for k,v in state.items() if k.startswith('backbone.')}
    if not backbone:
        backbone = state
    model.load_state_dict(backbone, strict=True, assign=True)
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Checkpoint contains non-finite backbone values')
    model.to(device=torch.device(device),dtype=torch.float32).eval().requires_grad_(False)
    return model


def normalize_uint8(volume):
    """Input axes are raw NIfTI array axes [H,W,D], matching DINO training."""
    array = np.asarray(volume)
    if array.ndim != 3 or array.dtype != np.uint8:
        raise ValueError('Provide a 3D uint8 volume; preprocess other intensity scales explicitly')
    return torch.from_numpy(array.astype(np.float32))[None]/127.5-1.


def make_feature_crops(volume, centers, *, output_size=128, global_size=96,
                       seed=42, case_id='case'):
    if volume.ndim!=4 or volume.shape[0]!=1 or not volume.numel() or not torch.isfinite(volume).all():
        raise ValueError('Expected finite normalized [1,H,W,D] volume')
    centers=np.asarray(centers,dtype=np.float64)
    if centers.ndim!=2 or centers.shape[1]!=3 or not len(centers) or not np.isfinite(centers).all():
        raise ValueError('Expected nonempty finite [N,3] raw voxel centers')
    if output_size<=0 or output_size%16:
        raise ValueError('Feature crop size must be a positive multiple of ViT patch size 16')
    if global_size<1 or seed<0:raise ValueError('Use a positive global crop size and nonnegative seed')
    shape=volume.shape[1:]
    salt=int.from_bytes(hashlib.sha256(str(case_id).encode()).digest()[:4],'little')
    rng=np.random.default_rng(seed+salt)
    crops=[]
    for _ in range(2):
        center=_random_global_center(rng,*shape,global_size)
        crop=_centered_crop(volume,(global_size,)*3,center)
        crops.append(F.interpolate(crop[None],size=(output_size,)*3,
                                   mode='trilinear',align_corners=False)[0])
    # Stable order; repeat available proposals for tiny/deduplicated volumes.
    for i in range(8):
        center=np.rint(np.clip(centers[i%len(centers)],0,np.array(shape)-1)).astype(int)
        crops.append(_centered_crop(volume,(output_size,)*3,tuple(center)))
    return torch.stack(crops)


@torch.inference_mode()
def encode_crops(model,crops,*,batch_size=1,return_patch_tokens=False):
    if batch_size<1 or crops.ndim!=5 or crops.shape[1]!=1 or not crops.numel() or not crops.is_floating_point() or not torch.isfinite(crops).all():
        raise ValueError('Expected [N,1,H,W,D] crops and a positive batch size')
    if any(s%model.patch_size for s in crops.shape[2:]):
        raise ValueError('All crop dimensions must be multiples of the patch size')
    device=next(model.parameters()).device
    cls,patches=[],[]
    for x in crops.split(batch_size):
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            out=model.forward_features(x.to(device))
        cls.append(out['x_norm_clstoken'].float().cpu())
        if return_patch_tokens:patches.append(out['x_norm_patchtokens'].float().cpu())
    result={'cls':torch.cat(cls)}
    if return_patch_tokens:result['patch_tokens']=torch.cat(patches)
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise RuntimeError('Feature extraction produced non-finite values')
    return result
