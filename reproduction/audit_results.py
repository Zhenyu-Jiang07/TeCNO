"""Audit a completed run against original labels, using independent sklearn metrics."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from .common import ROOT, PHASES, atomic_json, config, digest, ids, now


def audit(c):
    run=Path(c['run_root']);processed=Path(c['processed_root']);features=Path(c['features_root'])
    result=json.loads((run/'evaluation/test/results.json').read_text())
    assert result['signature']['videos']==ids(c,'test')
    assert result['signature']['limit'] is None
    feature_provenance=json.loads((features/'provenance.json').read_text())
    assert feature_provenance['limit'] is None and feature_provenance['fps']==5
    assert feature_provenance['encoder_train_videos']==ids(c,'train')
    feature_rows=0
    for v in range(1,81):
        df=pd.read_csv(processed/'manifests'/f'video{v:02d}.csv')
        fpath=features/f'video{v:02d}.npz'
        meta=json.loads((features/f'video{v:02d}.json').read_text())
        assert digest(fpath)==meta['sha256']
        with np.load(fpath,allow_pickle=False) as f:
            assert f['features'].shape==(len(df),2048)
            np.testing.assert_array_equal(f['frame'],df.frame.to_numpy())
            np.testing.assert_array_equal(f['phase'],df.phase.to_numpy())
            assert f['encoder_sha256'].item()==feature_provenance['encoder_sha256']
        feature_rows+=len(df)
    assert feature_rows==922570
    methods={name:{'acc':[],'p':[],'r':[],'correct':0} for name in ['cnn','tecno']}
    video_rows=[];test_frames=0
    for v in ids(c,'test'):
        df=pd.read_csv(run/'evaluation/test'/f'video{v:02d}.csv')
        source=pd.read_csv(Path(c['raw_root'])/'phase_annotations'/f'video{v:02d}-phase.txt',sep=r'\s+')
        expected=pd.read_csv(processed/'manifests'/f'video{v:02d}.csv')
        np.testing.assert_array_equal(df.frame,expected.frame)
        direct_truth=source.Phase.iloc[df.frame.to_numpy()].map({p:i for i,p in enumerate(PHASES)}).to_numpy()
        np.testing.assert_array_equal(df.ground_truth,direct_truth)
        truth=df.ground_truth.to_numpy();test_frames+=len(df)
        row={'video':v,'frames':len(df),'gt_transitions':int(np.count_nonzero(truth[1:]!=truth[:-1]))}
        for method,column in [('cnn','cnn_prediction'),('tecno','tecno_prediction')]:
            pred=df[column].to_numpy()
            p,r,_,_=precision_recall_fscore_support(truth,pred,labels=list(range(7)),zero_division=np.nan)
            acc=accuracy_score(truth,pred)
            methods[method]['acc'].append(acc);methods[method]['p'].append(p);methods[method]['r'].append(r)
            methods[method]['correct']+=int(np.count_nonzero(truth==pred))
            row[f'{method}_accuracy']=acc
            row[f'{method}_transitions']=int(np.count_nonzero(pred[1:]!=pred[:-1]))
        video_rows.append(row)
    assert test_frames==383817
    independent={};phase_table=pd.DataFrame({'phase_id':range(7),'phase':PHASES})
    for method,values in methods.items():
        p=np.nanmean(np.asarray(values['p']),axis=0);r=np.nanmean(np.asarray(values['r']),axis=0)
        calculated={'accuracy_video_mean':float(np.mean(values['acc'])),
                    'accuracy_frame_pooled':values['correct']/test_frames,
                    'precision_phase_then_video_macro':float(np.nanmean(p)),
                    'recall_phase_then_video_macro':float(np.nanmean(r))}
        for key,value in calculated.items():np.testing.assert_allclose(value,result[method][key],rtol=0,atol=1e-12)
        np.testing.assert_allclose(p,np.asarray(result[method]['precision_per_phase'],float),equal_nan=True)
        np.testing.assert_allclose(r,np.asarray(result[method]['recall_per_phase'],float),equal_nan=True)
        independent[method]=calculated
        phase_table[f'{method}_precision']=p;phase_table[f'{method}_recall']=r
    checks={};fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,stage in zip(axes,['cnn','tcn']):
        done=json.loads((run/stage/'completed.json').read_text())
        assert done['epochs_completed']==c[stage]['max_epochs']
        assert done['training_videos']==ids(c,'train') and done['validation_videos']==ids(c,'val')
        state=torch.load(done['last_checkpoint'],map_location='cpu',weights_only=False)
        assert state['epoch']+1==c[stage]['max_epochs'] and state['optimizer_states']
        expected_steps=15*((431560+63)//64) if stage=='cnn' else 25*40
        assert state['global_step']==expected_steps
        assert all(torch.isfinite(t).all() for t in state['state_dict'].values() if t.is_floating_point())
        checks[stage]={'epochs':done['epochs_completed'],'global_step':state['global_step'],
                       'best_checkpoint':done['best_checkpoint'],'best_sha256':digest(done['best_checkpoint'])}
        del state
        best=torch.load(done['best_checkpoint'],map_location='cpu',weights_only=False)
        checks[stage]['best_epoch_1based']=best['epoch']+1
        del best
        history=[json.loads(line) for line in (run/stage/'epochs.jsonl').read_text().splitlines()]
        rows={r['epoch']:r['validation']['accuracy_video_mean']*100 for r in history}
        assert sorted(rows)==list(range(c[stage]['max_epochs']))
        ax.plot([e+1 for e in sorted(rows)],[rows[e] for e in sorted(rows)],marker='o',markersize=3)
        e=checks[stage]['best_epoch_1based']
        ax.scatter([e],[rows[e-1]],color='#bd3c32',zorder=3,label=f'Selected epoch {e}')
        ax.set(title=f'{stage.upper()} validation',xlabel='Epoch',ylabel='Mean video accuracy (%)')
        ax.grid(alpha=.2);ax.legend()
    assert checks['cnn']['best_sha256']==feature_provenance['encoder_sha256']
    assert checks['tcn']['best_sha256']==result['signature']['checkpoint_sha256']
    assert digest(run/'tcn/features-provenance.json')==digest(features/'provenance.json')
    assert digest(run/'cnn/training_statistics.json')==digest(processed/'training_statistics.json')
    fig.tight_layout();fig.savefig(ROOT/'reports/validation-curves.png',dpi=180)
    fig.savefig(ROOT/'reports/validation-curves.pdf');plt.close(fig)
    video_table=pd.DataFrame(video_rows);video_table.to_csv(ROOT/'reports/test-per-video.csv',index=False)
    phase_table.to_csv(ROOT/'reports/test-per-phase.csv',index=False)
    # Diagnostic selection after evaluation; never used to select a checkpoint.
    from .evaluate import timeline
    worst=int(video_table.loc[video_table.tecno_accuracy.idxmin(),'video'])
    example=pd.read_csv(run/'evaluation/test'/f'video{worst:02d}.csv')
    timeline(example.frame.to_numpy(),example.ground_truth.to_numpy(),
             example.cnn_prediction.to_numpy(),example.tecno_prediction.to_numpy(),
             run/'evaluation/test'/f'video{worst:02d}-error-timeline',
             title=f'Cholec80 video{worst:02d} (test, 5 fps; error analysis)')
    output={'status':'completed_run_independently_verified','time':now(),'seed':c['seed'],
            'feature_videos':80,'feature_frames':feature_rows,'test_videos':32,'test_frames':test_frames,
            'ground_truth_verified_against_original_annotations':True,'sklearn_recomputed_metrics':independent,
            'checkpoints':checks,'total_transitions':{k:int(video_table[k].sum()) for k in ['gt_transitions','cnn_transitions','tecno_transitions']},
            'accuracy_difference_vs_paper_percentage_points':100*(independent['tecno']['accuracy_video_mean']-.8856),
            'scope':'single seed, strict 5fps frames, 40/8/32 split; not a five-run statistical replication'}
    atomic_json(ROOT/'reports/verified-results.json',output)
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/reproduction/cholec80.yaml')
    audit(config(ap.parse_args().config))
