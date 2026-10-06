import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch

from .common import PHASES,atomic_json,config,digest,ids,now,seed_all
from .data import VideoDataset
from .extract import load_model
from .metrics import confusion,summarize
from .train import require_free_gpu


def timeline(frame,truth,cnn,pred,out,fps=25,title=''):
    colors=['#6c8ebf','#82b366','#d6b656','#b85450','#9673a6','#d79b00','#67ab9f']
    fig,ax=plt.subplots(figsize=(15,3.7))
    ax.imshow(np.stack([truth,cnn,pred]),aspect='auto',interpolation='nearest',
              cmap=ListedColormap(colors),vmin=0,vmax=6,
              extent=[frame[0]/fps/60,(frame[-1]+(frame[1]-frame[0] if len(frame)>1 else 1))/fps/60,2.5,-.5])
    ax.set_yticks([0,1,2],['Ground truth','ResNet50','TeCNO']);ax.set_xlabel('Time (minutes)');ax.set_title(title)
    for boundary in [.5,1.5]:ax.axhline(boundary,color='white',linewidth=2)
    ax.legend(handles=[Patch(color=colors[i],label=f'{i+1}. {p}') for i,p in enumerate(PHASES)],
              loc='upper center',bbox_to_anchor=(.5,-.30),ncol=3,frameon=False,fontsize=9)
    fig.subplots_adjust(bottom=.38,left=.10,right=.99,top=.88)
    fig.savefig(out.with_suffix('.png'),dpi=180);fig.savefig(out.with_suffix('.pdf'));plt.close(fig)


def evaluate(c,checkpoint,split='test',limit=None):
    require_free_gpu();seed_all(c['seed'])
    out=Path(c['run_root'])/'evaluation'/split;out.mkdir(parents=True,exist_ok=True)
    signature={'checkpoint_sha256':digest(checkpoint),'split':split,'videos':ids(c,split),
               'features_provenance_sha256':digest(Path(c['features_root'])/'provenance.json'),'limit':limit}
    feature_snapshot=Path(checkpoint).parent.parent/'features-provenance.json'
    assert feature_snapshot.exists(),'Missing temporal-model training feature provenance'
    assert digest(feature_snapshot)==signature['features_provenance_sha256'], 'TCN inference features differ from its training encoder'
    if (out/'results.json').exists():
        prior=json.loads((out/'results.json').read_text())
        assert prior['signature']==signature,'Refusing to overwrite evaluation of another checkpoint'
        return prior
    model=load_model(checkpoint,c,'tcn')
    ds=VideoDataset(c,split,limit=limit);cms=[];cnn_cms=[];rows=[]
    for i in range(len(ds)):
        x,y,v,frame=ds[i]
        with torch.inference_mode():pred=model(x[None].cuda())[-1,0].argmax(0).cpu().numpy()
        truth=y.numpy()
        with np.load(Path(c['features_root'])/f'video{v:02d}.npz') as f:
            cnn=f['phase_logits'][:len(truth)].argmax(1)
        cm=confusion(truth,pred);cms.append(cm);cnn_cms.append(confusion(truth,cnn))
        pd.DataFrame({'frame':frame,'seconds':frame/25,'ground_truth':truth,'cnn_prediction':cnn,'tecno_prediction':pred}).to_csv(out/f'video{v:02d}.csv',index=False)
        rows.append({'video':v,'frames':len(y),'accuracy':float((pred==truth).mean()),'confusion':cm.tolist()})
        if i==0:
            timeline(frame,truth,cnn,pred,out/f'video{v:02d}-timeline',title=f'Cholec80 video{v:02d} ({split}, {c["fps"]} fps)')
    result={'status':'evaluated','time':now(),'signature':signature,'checkpoint':str(checkpoint),
            'tecno':summarize(cms),'cnn':summarize(cnn_cms),'per_video':rows,
            'protocol':'strict sampled frames; no relaxed boundary tolerance; no smoothing/postprocessing',
            'paper_arxiv_v1':{'accuracy':.8856,'precision':.8164,'recall':.8524,'runs':5},
            'comparison_limits':['This experiment is one seed; paper SD is across five runs.',
              'Published metric aggregation is underspecified; both video mean and pooled accuracy are reported.',
              'See REPRODUCTION.md for code, preprocessing and training differences.']}
    atomic_json(out/'results.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/reproduction/cholec80.yaml')
    ap.add_argument('--checkpoint');ap.add_argument('--split',choices=['val','test'],default='test')
    ap.add_argument('--limit',type=int);args=ap.parse_args();c=config(args.config)
    checkpoint=args.checkpoint or json.loads((Path(c['run_root'])/'tcn/completed.json').read_text())['best_checkpoint']
    evaluate(c,checkpoint,args.split,args.limit)


if __name__=='__main__':main()
