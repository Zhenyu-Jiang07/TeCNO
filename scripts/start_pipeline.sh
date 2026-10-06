#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export TORCH_HOME="$PWD/data/pretrained"
/data/users/jiangzhenyu/miniconda3/envs/tecno/bin/python - <<'PY'
import pathlib,subprocess
root=pathlib.Path.cwd()
pidfile=root/'logs/pipeline.pid'
if pidfile.exists() and pathlib.Path(f'/proc/{pidfile.read_text().strip()}/cmdline').exists():
    cmd=pathlib.Path(f'/proc/{pidfile.read_text().strip()}/cmdline').read_bytes()
    if b'reproduction.pipeline' in cmd:
        print('Pipeline already running:',pidfile.read_text().strip());raise SystemExit(0)
with open(root/'logs/pipeline.log','ab',buffering=0) as log:
    p=subprocess.Popen(['/data/users/jiangzhenyu/miniconda3/envs/tecno/bin/python','-u','-m','reproduction.pipeline'],
        cwd=root,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
pidfile.write_text(str(p.pid)+'\n')
print('Detached pipeline PID:',p.pid)
PY
