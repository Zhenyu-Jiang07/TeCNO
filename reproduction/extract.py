import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import atomic_json,config,digest,ids,now,seed_all
from .data import FrameDataset
from .lightning import TeCNORuntime
from .train import require_free_gpu


def load_model(checkpoint,c,stage,device='cuda'):
    # Only locally generated, trusted checkpoints. They include optimizer and RNG state.
    p=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert p['hyper_parameters']['stage']==stage
    assert p['hyper_parameters']['c']==c,'Checkpoint configuration mismatch'
    model=TeCNORuntime(c,stage,p['hyper_parameters']['weights'],initialize_pretrained=False)
    model.load_state_dict(p['state_dict'],strict=True)
    return model.to(device).eval()


def extract(c,checkpoint,videos=None,batch_size=128,limit=None):
    require_free_gpu();seed_all(c['seed'])
    videos=videos or list(range(1,81));out=Path(c['features_root']);out.mkdir(parents=True,exist_ok=True)
    ckpt_hash=digest(checkpoint)
    provenance={'encoder_checkpoint':str(Path(checkpoint).resolve()),'encoder_sha256':ckpt_hash,
                'fps':c['fps'],'dimension':2048,'dtype':'float32','encoder_train_videos':ids(c,'train'),
                'encoder_validation_videos':ids(c,'val'),'statistics_sha256':digest(Path(c['processed_root'])/'training_statistics.json'),
                'augmentation':False,'inference_mode':'eval','limit':limit}
    path=out/'provenance.json'
    if path.exists():assert json.loads(path.read_text())==provenance,'Feature cache belongs to a different encoder/protocol'
    atomic_json(path,provenance)
    model=load_model(checkpoint,c,'cnn')
    for v in videos:
        dest=out/f'video{v:02d}.npz'
        done=out/f'video{v:02d}.json'
        if dest.exists() and done.exists():
            previous=json.loads(done.read_text())
            if previous['encoder_sha256']==ckpt_hash and previous['sha256']==digest(dest):
                print(f'cached video{v:02d}',flush=True);continue
        ds=FrameDataset(c,videos=[v],limit=limit)
        dl=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=c['num_workers'],pin_memory=True)
        feats=[];phase_logits=[];truth=[];frames=[]
        with torch.inference_mode():
            for x,y,_,video,frame in dl:
                stem,p,_=model(x.cuda(non_blocking=True))
                feats.append(stem.cpu().numpy());phase_logits.append(p.cpu().numpy())
                truth.append(y.numpy());frames.append(frame.numpy())
        f=np.concatenate(feats).astype(np.float32);p=np.concatenate(phase_logits).astype(np.float32)
        y=np.concatenate(truth).astype(np.int64);frame=np.concatenate(frames).astype(np.int64)
        assert f.shape==(len(ds),2048) and np.isfinite(f).all() and np.isfinite(p).all()
        assert np.array_equal(frame,ds.frames)
        with open(dest.with_suffix('.part'),'wb') as handle:
            np.savez(handle,features=f,phase_logits=p,phase=y,frame=frame,
                     encoder_sha256=np.array(ckpt_hash),fps=np.array(c['fps']))
        dest.with_suffix('.part').replace(dest)
        atomic_json(done,{'video':v,'frames':len(y),'encoder_sha256':ckpt_hash,'sha256':digest(dest),'time':now()})
        print(f'extracted features video{v:02d} {f.shape}',flush=True)
    return provenance


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/reproduction/cholec80.yaml')
    ap.add_argument('--checkpoint');ap.add_argument('--videos',type=int,nargs='+');ap.add_argument('--limit',type=int)
    args=ap.parse_args();c=config(args.config)
    ckpt=args.checkpoint or json.loads((Path(c['run_root'])/'cnn/completed.json').read_text())['best_checkpoint']
    extract(c,ckpt,args.videos,limit=args.limit)


if __name__=='__main__':main()
