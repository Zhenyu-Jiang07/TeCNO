"""Real-case pipeline test, isolated artifacts, never a test-set result."""
import json
from pathlib import Path
import subprocess
import sys
import yaml

from .common import ROOT,atomic_json,config,digest,now
from .prepare import make_video,normalization


def main():
    import torch
    base=config('configs/reproduction/cholec80.yaml')
    c=dict(base);c['train_videos']=[36,36];c['val_videos']=[41,41]
    c['run_root']=str(ROOT/'runs/smoke-real');c['features_root']=str(ROOT/'features/smoke-real')
    c['processed_root']=str(ROOT/'data/processed/smoke-real');c['num_workers']=2
    c['experiment']='smoke_only_video36_train_video41_val'
    for v in [36,41]:make_video(base,v)
    dest=Path(c['processed_root']);dest.mkdir(parents=True,exist_ok=True)
    for part in ['frames','manifests']:
        target=dest/part
        if not target.exists():target.symlink_to(Path(base['processed_root'])/part,target_is_directory=True)
    if not (dest/'training_statistics.json').exists():normalization(c)
    out=Path(c['run_root']);out.mkdir(parents=True,exist_ok=True)
    cfg=out/'smoke.yaml';cfg.write_text(yaml.safe_dump(c))
    inference_cfg=out/'inference.yaml'
    inference_cfg.write_text(yaml.safe_dump({**c,'run_root':str(out/'smoke')}))
    steps=[]
    def call(module,*args):
        command=[sys.executable,'-u','-m',module,'--config',str(inference_cfg),*args]
        # Each GPU stage runs in its own process, releasing CUDA before the next.
        subprocess.run(command,check=True)
    for stage in ['cnn']:
        for epochs in [1,2]:
            last=out/f'smoke/{stage}/checkpoints/last.ckpt'
            if last.exists() and torch.load(last,map_location='cpu',weights_only=False)['epoch']+1>=epochs:
                continue
            command=[sys.executable,'-u','-m','reproduction.train',stage,'--config',str(cfg),'--smoke','--epochs',str(epochs)]
            subprocess.run(command,check=True)
    cnn=json.loads((out/'smoke/cnn/completed.json').read_text())
    call('reproduction.extract','--checkpoint',cnn['best_checkpoint'],'--videos','36','41','--limit','256')
    # Cache reuse must not alter feature files.
    first=digest(Path(c['features_root'])/'video36.npz')
    call('reproduction.extract','--checkpoint',cnn['best_checkpoint'],'--videos','36','41','--limit','256')
    assert digest(Path(c['features_root'])/'video36.npz')==first
    for epochs in [1,2]:
        last=out/'smoke/tcn/checkpoints/last.ckpt'
        if last.exists() and torch.load(last,map_location='cpu',weights_only=False)['epoch']+1>=epochs:
            continue
        subprocess.run([sys.executable,'-u','-m','reproduction.train','tcn','--config',str(cfg),'--smoke','--epochs',str(epochs)],check=True)
    tcn=json.loads((out/'smoke/tcn/completed.json').read_text())
    call('reproduction.evaluate','--checkpoint',tcn['best_checkpoint'],'--split','val','--limit','256')
    states={}
    for stage in ['cnn','tcn']:
        state=torch.load(out/f'smoke/{stage}/checkpoints/last.ckpt',map_location='cpu',weights_only=False)
        assert state['epoch']==1 and state['global_step']>=2 and state['optimizer_states']
        states[stage]={'epoch':state['epoch'],'global_step':state['global_step'],'optimizer_restored':True}
    atomic_json(out/'completed.json',{'status':'real_data_smoke_passed','time':now(),
                'checkpoint_checks':states,'feature_dimension':2048,'cache_reuse_verified':True,
                'train_case':36,'validation_case':41,'test_used':False,
                'warning':'Only a pipeline test. Tiny subsets and two short epochs; not reproduction performance.'})


if __name__=='__main__':main()
