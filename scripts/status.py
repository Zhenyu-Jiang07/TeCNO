"""Read-only compact status: python scripts/status.py."""
import json
from pathlib import Path
import yaml

root=Path(__file__).resolve().parents[1]
c=yaml.safe_load((root/'configs/reproduction/cholec80.yaml').read_text())
p=root/'logs/pipeline-status.json'
if not p.exists():
    print('Pipeline has not started.')
    raise SystemExit(0)
s=json.loads(p.read_text())
stage=s.get('active_stage')
parts=[s.get('time',''),s['status'],f'stage={stage}',
       f"data={len(s.get('prepared_videos',[]))}/80"]
if stage in ('cnn','tcn'):
    base=root/c['run_root']/stage
    if (base/'status.json').exists():
        t=json.loads((base/'status.json').read_text())
        parts += [f"epoch={t.get('epoch',-1)+1}/{c[stage]['max_epochs']}",
                  f"batch={t.get('batch')}",f"step={t.get('step')}"]
    if (base/'epochs.jsonl').exists():
        history=[]
        for line in (base/'epochs.jsonl').read_text().splitlines():
            try:history.append(json.loads(line))
            except json.JSONDecodeError:pass
        if history:
            parts.append(f"val_latest={history[-1]['validation']['accuracy_video_mean']*100:.2f}%")
            parts.append(f"val_best={max(x['validation']['accuracy_video_mean'] for x in history)*100:.2f}%")
if stage=='features':parts.append(f"features={len(list((root/c['features_root']).glob('video*.json')))}/80")
if s['status']=='completed':
    result_path=root/c['run_root']/'evaluation/test/results.json'
    if result_path.exists():
        metrics=json.loads(result_path.read_text())['tecno']
        parts.append(f"test_acc={100*metrics['accuracy_video_mean']:.2f}%")
        parts.append(f"test_precision={100*metrics['precision_phase_then_video_macro']:.2f}%")
        parts.append(f"test_recall={100*metrics['recall_phase_then_video_macro']:.2f}%")
if s.get('error'):parts.append(s['error'])
parts.append(f"retries={s.get('attempts',{})}")
print(' | '.join(parts))
