#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export HF_TOKEN="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"
export SAM3_CHUNK_FRAMES="${SAM3_CHUNK_FRAMES:-100}"
exec /kaggle/working/miniforge3/envs/sam3/bin/python /kaggle/working/SAM3/sam3_remote_pipeline.py "$@"
