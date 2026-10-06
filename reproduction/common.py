import hashlib
import json
from pathlib import Path
import random
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
PHASES = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting',
          'GallbladderDissection', 'GallbladderPackaging', 'CleaningCoagulation',
          'GallbladderRetraction']
TOOLS = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'SpecimenBag']


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def config(path):
    c = yaml.safe_load(Path(path).read_text())
    for key in ['raw_root', 'processed_root', 'features_root', 'run_root']:
        c[key] = str((ROOT / c[key]).resolve())
    sets = [set(range(c[k][0], c[k][1]+1)) for k in ['train_videos', 'val_videos', 'test_videos']]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i)), 'Case leakage'
    assert c['source_fps'] % c['fps'] == 0
    return c


def ids(c, split):
    a, b = c[f'{split}_videos']
    return list(range(a, b+1))


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    # Explicitly document numerical choices; no unrecorded TF32 acceleration.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def phase_weights(labels):
    counts = np.bincount(labels, minlength=7)
    assert (counts > 0).all(), f'Missing training phase: {counts}'
    return (np.median(counts) / counts).astype(np.float32)
