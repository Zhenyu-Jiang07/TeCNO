"""Durable local supervisor: download/preprocess overlap, checkpointed GPU stages.

Only this project's recorded processes are managed. No process is terminated.
Failure is persisted; no failed/incomplete run is labelled as reproduced.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import ROOT,atomic_json,config,now


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:return False


class RecordedProcess:
    """Adopt a detached child after a supervisor restart without duplicating it."""
    def __init__(self,pid,done):self.pid=pid;self.done=done;self.returncode=None
    def poll(self):
        if alive(self.pid):return None
        self.returncode=0 if self.done() else 1
        return self.returncode


def background(command,logname):
    (ROOT/'logs').mkdir(parents=True,exist_ok=True)
    with open(ROOT/'logs'/f'{logname}.log','ab',buffering=0) as log:
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                           start_new_session=True)
    (ROOT/'logs'/f'{logname}.pid').write_text(str(p.pid)+'\n')
    return p


def main():
    os.chdir(ROOT)
    (ROOT/'logs').mkdir(parents=True,exist_ok=True)
    lock=open(ROOT/'logs/pipeline.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/'logs/pipeline.pid').write_text(str(os.getpid())+'\n')
    c=config('configs/reproduction/cholec80.yaml');run=Path(c['run_root'])
    py=sys.executable;prep_jobs={};attempts={};active=None;active_stage=None
    stages=[('cnn',[py,'-u','-m','reproduction.train','cnn']),
            ('features',[py,'-u','-m','reproduction.extract']),
            ('tcn',[py,'-u','-m','reproduction.train','tcn']),
            ('evaluation',[py,'-u','-m','reproduction.evaluate','--split','test'])]
    completed={}
    for stage,path in [('cnn',run/'cnn/completed.json'),('tcn',run/'tcn/completed.json'),
                       ('evaluation',run/'evaluation/test/results.json')]:
        completed[stage]=path.exists()
    completed['features']=all((Path(c['features_root'])/f'video{v:02d}.json').exists() for v in range(1,81))
    status_path=ROOT/'logs/pipeline-status.json'
    if status_path.exists():
        previous=json.loads(status_path.read_text())
        previous_pid=previous.get('active_pid');previous_stage=previous.get('active_stage')
        if previous_pid and alive(previous_pid):
            active_stage=previous_stage
            def finished(stage=previous_stage):
                if stage=='statistics':return (Path(c['processed_root'])/'training_statistics.json').exists()
                if stage=='features':return all((Path(c['features_root'])/f'video{v:02d}.json').exists() for v in range(1,81))
                if stage=='evaluation':return (run/'evaluation/test/results.json').exists()
                return (run/stage/'completed.json').exists()
            active=RecordedProcess(previous_pid,finished)
    for v in range(1,81):
        pf=ROOT/'logs'/f'prepare-video{v:02d}.pid'
        if pf.exists() and alive(int(pf.read_text())):
            prep_jobs[v]=RecordedProcess(int(pf.read_text()),
                        lambda v=v:(Path(c['processed_root'])/'audit'/f'video{v:02d}.json').exists())
    for tick in range(10**9):
        # Restart only the known project transfer/unpack workers if needed.
        aux=[('download',ROOT/'data/raw/cholec80.chunks/COMPLETE',[py,'-u','scripts/download_cholec80.py']),
             ('unpack',Path(c['raw_root'])/'complete.json',[py,'-u','-c',
              'from reproduction.prepare import unpack;from reproduction.common import config;unpack(config("configs/reproduction/cholec80.yaml"),True)'])]
        for name,done,cmd in aux:
            pidfile=ROOT/'logs'/f'{name}.pid'
            running=pidfile.exists() and alive(int(pidfile.read_text()))
            if not done.exists() and not running:
                attempts[name]=attempts.get(name,0)+1
                if attempts[name]>3:raise RuntimeError(f'{name} failed repeatedly; inspect logs/{name}.log')
                background(cmd,name)
        for v,p in list(prep_jobs.items()):
            code=p.poll()
            if code is not None:
                del prep_jobs[v]
                if code:raise RuntimeError(f'Preprocessing video{v:02d} failed: logs/prepare-video{v:02d}.log')
        for v in range(1,81):
            if len(prep_jobs)>=3:break
            ready=(Path(c['raw_root'])/'videos'/f'video{v:02d}.mp4.verified.json').exists()
            done=(Path(c['processed_root'])/'audit'/f'video{v:02d}.json').exists()
            if ready and not done and v not in prep_jobs:
                # The interactive real-data smoke worker prepares these two cases.
                smoke_pid=ROOT/'logs/smoke-real.pid'
                if v in [36,41] and smoke_pid.exists() and alive(int(smoke_pid.read_text())):continue
                prep_jobs[v]=background([py,'-u','-m','reproduction.prepare','--videos',str(v),'--workers','1'],f'prepare-video{v:02d}')
        prepared=[v for v in range(1,81) if (Path(c['processed_root'])/'audit'/f'video{v:02d}.json').exists()]
        smoke_done=ROOT/'runs/smoke-real/completed.json'
        smoke_pidfile=ROOT/'logs/smoke-real.pid'
        smoke_running=smoke_pidfile.exists() and alive(int(smoke_pidfile.read_text()))
        if not smoke_done.exists() and not smoke_running and all(v in prepared for v in [36,41]):
            occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                        '--format=csv,noheader'],text=True).strip()
            if not occupied:
                attempts['smoke']=attempts.get('smoke',0)+1
                if attempts['smoke']>3:raise RuntimeError('Real-data smoke repeatedly failed; inspect logs/smoke-real.log')
                background([py,'-u','-m','reproduction.smoke'],'smoke-real')
        chunks=len(list((ROOT/'data/raw/cholec80.chunks').glob('[0-9]*.json')))
        if active is not None and active.poll() is not None:
            code=active.returncode
            if code:
                attempts[active_stage]=attempts.get(active_stage,0)+1
                log=(ROOT/'logs'/f'formal-{active_stage}.log').read_text(errors='replace')[-8000:]
                if 'GPU occupied by a compute process' in log:
                    attempts[active_stage]-=1
                elif attempts[active_stage]>=3:
                    raise RuntimeError(f'{active_stage} failed three times; see logs/formal-{active_stage}.log')
            else:completed[active_stage]=True
            active=None;active_stage=None
        if active is None:
            if (ROOT/'runs/smoke-real/completed.json').exists() and all(v in prepared for v in range(1,49)):
                statistics=Path(c['processed_root'])/'training_statistics.json'
                if not statistics.exists() and not completed.get('statistics'):
                    active=background([py,'-u','-c','from reproduction.prepare import normalization;from reproduction.common import config;normalization(config("configs/reproduction/cholec80.yaml"))'],'formal-statistics')
                    active_stage='statistics'
                else:
                    for stage,command in stages:
                        if completed[stage]:continue
                        if stage=='features' and (len(prepared)!=80 or
                            not (ROOT/'data/raw/cholec80.chunks/COMPLETE').exists() or
                            not (Path(c['raw_root'])/'complete.json').exists()):break
                        # Wait cooperatively if someone else has started a compute task.
                        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                    '--format=csv,noheader'],text=True).strip()
                        if occupied:break
                        active=background(command,f'formal-{stage}');active_stage=stage
                        break
        status={'time':now(),'status':'running','active_stage':active_stage,'active_pid':active.pid if active else None,
                'download_chunks_complete':chunks,'download_chunks_total':2233,'prepared_videos':prepared,
                'preparing_videos':list(prep_jobs),'completed_stages':completed,
                'real_smoke_passed':(ROOT/'runs/smoke-real/completed.json').exists(),'attempts':attempts}
        if all(completed.get(k) for k,_ in stages):
            status['status']='completed'
            atomic_json(ROOT/'logs/pipeline-status.json',status)
            write_result(c)
            print(json.dumps(status),flush=True)
            return
        atomic_json(ROOT/'logs/pipeline-status.json',status)
        if tick%6==0:print(json.dumps(status),flush=True)
        time.sleep(10)


def write_result(c):
    run=Path(c['run_root']);r=json.loads((run/'evaluation/test/results.json').read_text());t=r['tecno'];cnn=r['cnn']
    text=f'''# TeCNO 单次复现实验结果

完成时间：{now()}。种子：{c['seed']}。训练 01–40，验证 41–48，测试 49–80；5 fps；CNN 阶段+器械监督。

| 指标 | 本次 TeCNO | 本次 CNN | 论文（五次均值±标准差） |
|---|---:|---:|---:|
| 逐视频平均 Acc | {t['accuracy_video_mean']*100:.2f}% | {cnn['accuracy_video_mean']*100:.2f}% | 88.56±0.27% |
| 全帧 Acc（补充） | {t['accuracy_frame_pooled']*100:.2f}% | {cnn['accuracy_frame_pooled']*100:.2f}% | 聚合方式未充分说明 |
| 宏 Precision | {t['precision_phase_then_video_macro']*100:.2f}% | {cnn['precision_phase_then_video_macro']*100:.2f}% | 81.64±0.41% |
| 宏 Recall | {t['recall_phase_then_video_macro']*100:.2f}% | {cnn['recall_phase_then_video_macro']*100:.2f}% | 85.24±1.06% |

与论文 Acc 均值的差为 {(t['accuracy_video_mean']-.8856)*100:+.2f} 个百分点。本次是完整单种子训练和严格帧评测，不是五次结果的统计复现。
论文评测聚合细节不完整；CNN 训练细节、现代依赖、训练统计和损失输入修正均见 REPRODUCTION.md，不能仅凭分数宣称方法完全等价或达到论文结果。
若有差距，优先核对这些实现/协议差异及随机种子影响；所有后续选择只能使用验证集。本次没有使用测试集选 checkpoint 或调参。

机器可读结果：`{run}/evaluation/test/results.json`。
测试视频 49 的真实/预测时间轴：`{run}/evaluation/test/video49-timeline.png`。
每视频预测和混淆矩阵保存在同一评测目录；训练配置、日志及 checkpoint 分别位于 `{run}/cnn` 和 `{run}/tcn`。
'''
    (ROOT/'reports/RESULTS.md').write_text(text)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic_json(ROOT/'logs/pipeline-status.json',{'status':'failed','time':now(),'error':repr(exc)})
        raise
