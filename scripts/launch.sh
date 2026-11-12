#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
stage=${1:-pretrain}
shift || true
case "$stage" in
  warmup) extra=(--warmup) ;;
  pretrain) extra=() ;;
  *) echo 'Usage: bash scripts/launch.sh {warmup|pretrain} [key=value ...]' >&2; exit 2 ;;
esac
cfg_opts=()
for arg in "$@"; do
  if [[ "$arg" =~ ^[A-Za-z_][A-Za-z0-9_.]*= ]]; then cfg_opts+=("$arg"); fi
done
python scripts/check_config.py --stage "$stage" --check-files "${cfg_opts[@]}"
exec torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-4}" \
  dinov3/train/train.py --config-file "configs/$stage.yaml" \
  --output-dir "${OUTPUT_BASE:-./outputs}/$stage" "${extra[@]}" "$@"
