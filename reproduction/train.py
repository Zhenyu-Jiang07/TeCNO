import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import tarfile

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger,TensorBoardLogger
import torch
from torch.utils.data import DataLoader

from .common import ROOT,atomic_json,config,ids,now,seed_all
from .data import FrameDataset,VideoDataset
from .lightning import TeCNORuntime


def require_free_gpu():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
                                   '--format=csv,noheader'],text=True).strip()
    if output:
        raise RuntimeError(f'GPU occupied by a compute process; retry later, never terminate it: {output}')


class Progress(Callback):
    def __init__(self,out):self.out=Path(out);self.started=time.monotonic()

    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        if batch_idx%100==0:
            loss=float(outputs['loss'].detach())
            if not torch.isfinite(torch.tensor(loss)):raise FloatingPointError(f'Nonfinite loss {loss}')
            data={'status':'training','time':now(),'stage':module.stage,'epoch':trainer.current_epoch,
                  'step':trainer.global_step,'batch':batch_idx,'loss':loss,
                  'elapsed_seconds':time.monotonic()-self.started,
                  'gpu_peak_GiB':torch.cuda.max_memory_allocated()/2**30}
            atomic_json(self.out/'status.json',data)
            print(json.dumps(data),flush=True)

    def on_validation_end(self,trainer,module):
        if trainer.sanity_checking:return
        data={'time':now(),'epoch':trainer.current_epoch,'step':trainer.global_step,
              'validation':module.latest_validation}
        with open(self.out/'epochs.jsonl','a') as f:f.write(json.dumps(data)+'\n')
        print(json.dumps(data),flush=True)


def run(c,stage,resume=True,smoke=False,max_epochs=None):
    require_free_gpu();seed_all(c['seed']);pl.seed_everything(c['seed'],workers=True)
    out=Path(c['run_root'])/stage
    out.mkdir(parents=True,exist_ok=True)
    last=out/'checkpoints/last.ckpt'
    saved=out/'config.json'
    if saved.exists():
        assert json.loads(saved.read_text())==c,'Configuration changed in existing run; use a different run directory'
    atomic_json(saved,c)
    snapshot=out/'source-snapshots'/now().replace(':','-')
    snapshot.mkdir(parents=True,exist_ok=True)
    with tarfile.open(snapshot/'source.tar.gz','w:gz') as archive:
        for folder in ['reproduction','models','configs/reproduction','scripts','tests']:
            for src in (ROOT/folder).rglob('*'):
                if src.is_file() and '__pycache__' not in src.parts:
                    archive.add(src,arcname=str(src.relative_to(ROOT)))
        for name in ['requirements-reproduction.txt','REPRODUCTION.md']:
            archive.add(ROOT/name,arcname=name)
    for filename in ['environment-lock.yml','pip-freeze.txt','nvidia-smi.txt']:
        if (ROOT/'reports'/filename).exists():shutil.copy2(ROOT/'reports'/filename,snapshot/filename)
    (out/'source.diff').write_text(subprocess.check_output(['git','diff'],cwd=ROOT,text=True))
    (out/'original_commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True))
    stats=json.loads((Path(c['processed_root'])/'training_statistics.json').read_text())
    assert stats['train_videos']==ids(c,'train')
    if stage=='cnn':
        train=FrameDataset(c,'train',augment=True,limit=32 if smoke else None)
        val=FrameDataset(c,'val',limit=32 if smoke else None)
    else:
        train=VideoDataset(c,'train',limit=256 if smoke else None)
        val=VideoDataset(c,'val',limit=256 if smoke else None)
    workers=0 if smoke else c['num_workers']
    kwargs={'batch_size':4 if smoke and stage=='cnn' else c[stage]['batch_size'],
            'num_workers':workers,'pin_memory':True,'persistent_workers':False}
    train_dl=DataLoader(train,shuffle=True,**kwargs)
    val_dl=DataLoader(val,shuffle=False,**kwargs)
    module=TeCNORuntime(c,stage,stats['class_weights'],initialize_pretrained=not (resume and last.exists()))
    ckpt=ModelCheckpoint(dirpath=out/'checkpoints',filename='epoch={epoch:02d}-val={val_acc_video:.5f}',
        auto_insert_metric_name=False,monitor=c[stage]['monitor'],mode='max',save_top_k=1,save_last=True,
        save_on_train_epoch_end=False)
    trainer=pl.Trainer(accelerator='gpu',devices=[0],precision=c[stage]['precision'],
        max_epochs=max_epochs or c[stage]['max_epochs'],deterministic=True,
        callbacks=[Progress(out),ckpt],logger=[CSVLogger(out,name='csv'),TensorBoardLogger(out,name='tensorboard')],
        num_sanity_val_steps=2,enable_progress_bar=False,log_every_n_steps=50,
        limit_train_batches=2 if smoke else 1.0,limit_val_batches=2 if smoke else 1.0,
        default_root_dir=out)
    trainer.fit(module,train_dataloaders=train_dl,val_dataloaders=val_dl,
                ckpt_path=str(last) if resume and last.exists() else None)
    result={'status':'smoke_completed' if smoke else 'training_completed','stage':stage,'time':now(),
            'best_checkpoint':ckpt.best_model_path,'best_validation_accuracy':float(ckpt.best_model_score),
            'last_checkpoint':str(last),'epochs_completed':trainer.current_epoch,
            'selection_split':'val','training_videos':ids(c,'train'),'validation_videos':ids(c,'val'),
            'test_evaluated':False}
    atomic_json(out/'completed.json',result);atomic_json(out/'status.json',result)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['cnn','tcn'])
    ap.add_argument('--config',default='configs/reproduction/cholec80.yaml')
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--epochs',type=int)
    args=ap.parse_args();c=config(args.config)
    if args.smoke:c['run_root']=str(Path(c['run_root'])/'smoke')
    print(run(c,args.stage,smoke=args.smoke,max_epochs=args.epochs))


if __name__=='__main__':main()
