#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONDA=/data/users/jiangzhenyu/miniconda3/bin/conda
TECNO_PY=/data/users/jiangzhenyu/miniconda3/envs/tecno/bin/python
if [[ ! -x "$TECNO_PY" ]]; then
    "$CONDA" create --name tecno python=3.11 pip --override-channels -c conda-forge -y
else
    echo 'Existing tecno environment: inspect pip list before making changes.'
    "$TECNO_PY" -m pip list
    exit 1
fi
"$TECNO_PY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$TECNO_PY" -m pip install -r requirements-reproduction.txt
"$CONDA" install --name tecno ffmpeg --override-channels -c conda-forge -y
