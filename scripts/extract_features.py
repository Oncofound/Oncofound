"""Raw NIfTI + raw proposal coordinates -> downstream-compatible .npy bags."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import nibabel as nib
import torch
from features.extractor import load_backbone,normalize_uint8,make_feature_crops,encode_crops


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proposals',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--config',default=str(ROOT/'configs/pretrain.yaml'))
    p.add_argument('--role',choices=['teacher','student'],default='teacher')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args(argv)
    if args.batch_size<1 or args.seed<0:raise ValueError('Use a positive batch size and nonnegative seed')
    out=Path(args.output_dir).absolute()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError('Feature output directory must be empty; choose a new directory')
    records=[];seen=set()
    for line in Path(args.proposals).read_text().splitlines():
        if not line.strip():continue
        record=json.loads(line);cid=record['case_id']
        if not isinstance(cid,str) or not cid.strip() or Path(cid).name!=cid or cid in seen or cid in ('.','..') or '\\' in cid or '\x00' in cid:
            raise ValueError('case_id must be unique and safe as a filename')
        if record.get('coord_space')!='raw_nifti_voxel':
            raise ValueError('Expected raw_nifti_voxel proposals from the converter')
        seen.add(cid);records.append(record)
    if not records:raise ValueError('Proposal file contains no cases')
    model=load_backbone(args.checkpoint,args.config,args.role,args.device)
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(dir=out.parent,prefix='.feature_extract_'))
    try:
        for record in records:
            volume=np.asarray(nib.load(record['image_path']).dataobj)
            if list(volume.shape)!=record['raw_shape']:raise ValueError('Raw proposal/image shape mismatch')
            crops=make_feature_crops(normalize_uint8(volume),record['proposal_centers'],seed=args.seed,case_id=record['case_id'])
            features=encode_crops(model,crops,batch_size=args.batch_size)['cls'].numpy()
            if features.shape!=(10,1024):raise ValueError('Unexpected encoder feature dimensions')
            np.save(stage/f"{record['case_id']}.npy",features,allow_pickle=False)
        def fingerprint(path):
            with open(path,'rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()
        report={'cases':len(records),'bag_shape':[10,1024],'checkpoint_role':args.role,'seed':args.seed,
                'checkpoint_sha256':fingerprint(args.checkpoint),'config_sha256':fingerprint(args.config)}
        (stage/'feature_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        # Publish the whole set once; a nonempty destination is never replaced.
        os.rename(stage,out)
    finally:
        if stage.exists():shutil.rmtree(stage)
    print(json.dumps(report))


if __name__=='__main__':main()
