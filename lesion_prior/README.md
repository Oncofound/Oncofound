# Local crop sampler

```bash
CROP_MANIFEST=./data/crop_manifest.csv bash scripts/launch_crop_light.sh
python scripts/select_local_crops.py --ckpt weights/sampler/best_all8_hit.pt --manifest data/cases.csv --output-dir outputs/proposals
```
