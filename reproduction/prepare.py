"""Prepare exactly frame 0,5,10,... at 5 fps, preserving original frame IDs."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np
import pandas as pd
from .common import ROOT, PHASES, TOOLS, atomic_json, config, digest, ids, now


def wait_archive_member(info, archive, wait):
    markers = archive.with_suffix('.chunks')
    chunk = 32 * 1024**2
    begin = info.header_offset // chunk
    end = min(archive.stat().st_size-1, info.header_offset + info.compress_size + len(info.filename.encode()) + 200) // chunk
    while not all((markers / f'{i:05d}.json').exists() for i in range(begin, end+1)):
        if not wait:
            return False
        time.sleep(10)
    return True


def unpack(c, wait=False):
    archive = ROOT / 'data/raw/cholec80.zip'
    size = 74916858781
    last = (size-1) // (32 * 1024**2)
    while not (archive.with_suffix('.chunks') / f'{last:05d}.json').exists():
        if not wait:
            raise RuntimeError('Archive central directory not downloaded')
        time.sleep(10)
    root = Path(c['raw_root'])
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        # Annotations first. Reading a member verifies its ZIP CRC32.
        def priority(x):
            if x.filename.startswith(('phase_annotations/', 'tool_annotations/')): return (0,x.header_offset)
            if x.filename == 'README.txt': return (1,x.header_offset)
            if x.filename.endswith('.mp4'): return (2,x.header_offset)
            return (3,x.header_offset)
        members = sorted(z.infolist(), key=priority)
        member_count=len(members)
        pending = [x for x in members if not x.is_dir()]
        while pending:
            remaining=[]
            for info in pending:
                dest = (root / info.filename).resolve()
                assert dest.is_relative_to(root.resolve()), 'Unsafe archive path'
                done = dest.with_suffix(dest.suffix+'.verified.json')
                if done.exists() and dest.exists() and dest.stat().st_size == info.file_size:
                    continue
                if not wait_archive_member(info, archive, False):
                    remaining.append(info)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix+'.part')
                import hashlib
                h = hashlib.sha256()
                with z.open(info) as src, open(tmp, 'wb') as dst:
                    while blob := src.read(8*1024**2):
                        dst.write(blob)
                        h.update(blob)
                assert tmp.stat().st_size == info.file_size
                tmp.replace(dest)
                atomic_json(done, {'member': info.filename, 'bytes': info.file_size,
                                   'crc32': info.CRC, 'sha256': h.hexdigest(), 'verified_at': now()})
                print(f'extracted {info.filename}', flush=True)
            if remaining and not wait:
                raise RuntimeError(f'{len(remaining)} archive members are still downloading')
            pending=remaining
            if pending:time.sleep(10)
    atomic_json(root/'complete.json', {'members': member_count, 'completed_at': now()})


def make_video(c, video):
    cv2.setNumThreads(1)
    raw, processed = Path(c['raw_root']), Path(c['processed_root'])
    vname = f'video{video:02d}'
    output = processed / 'frames' / vname
    manifest = processed / 'manifests' / f'{vname}.csv'
    audit = processed / 'audit' / f'{vname}.json'
    if audit.exists() and manifest.exists():
        a = json.loads(audit.read_text())
        assert a['fps'] == c['fps'] and a['storage_size'] == c['storage_size']
        if len(list(output.glob('*.png'))) == a['sampled_frames']:
            return a
    phase = pd.read_csv(raw/'phase_annotations'/f'{vname}-phase.txt', sep=r'\s+')
    tool = pd.read_csv(raw/'tool_annotations'/f'{vname}-tool.txt', sep=r'\s+')
    assert list(phase.columns) == ['Frame', 'Phase']
    assert list(tool.columns) == ['Frame'] + TOOLS
    assert phase.Frame.iloc[0] == 0 and tool.Frame.iloc[0] == 0
    assert np.array_equal(phase.Frame, np.arange(len(phase)))
    assert np.array_equal(tool.Frame, np.arange(len(tool))*25)
    assert phase.Phase.isin(PHASES).all() and tool[TOOLS].isin([0,1]).all().all()
    ffprobe = str(Path(sys.executable).parent/'ffprobe')
    ffmpeg = str(Path(sys.executable).parent/'ffmpeg')
    src = raw/'videos'/f'{vname}.mp4'
    probe = json.loads(subprocess.check_output([ffprobe,'-v','error','-select_streams','v:0',
                '-show_entries','stream=nb_frames,r_frame_rate,width,height,duration','-of','json',str(src)]))['streams'][0]
    assert probe['r_frame_rate'] in ('25/1', '50/2'), probe
    nvideo = int(probe['nb_frames'])
    # Allow a documented terminal annotation discrepancy, never an interior shift.
    assert abs(nvideo-len(phase)) <= 25, (vname,nvideo,len(phase))
    usable = min(nvideo,len(phase))
    stride = c['source_fps']//c['fps']
    frames = np.arange(0,usable,stride)
    tool_indices = np.searchsorted(tool.Frame.to_numpy(),frames,side='right')-1
    tool_age=frames-tool.Frame.to_numpy()[tool_indices]
    # Official files can include the terminal frame at exactly one second
    # after the final tool annotation; the author's dataframe repeats it once.
    assert (tool_age <= 25).all(), (vname, int(tool_age.max()))
    output.mkdir(parents=True,exist_ok=True)
    logpath = ROOT/'logs'/f'preprocess-{vname}.log'
    logpath.parent.mkdir(parents=True,exist_ok=True)
    with open(logpath,'a') as log:
        subprocess.run([ffmpeg,'-nostdin','-hide_banner','-loglevel','warning','-threads','2',
                        '-i',str(src),'-vf',f"select=not(mod(n\\,{stride}))*lt(n\\,{usable}),scale={c['storage_size']}:{c['storage_size']}",
                        '-fps_mode','passthrough','-start_number','0','-compression_level','3',
                        '-threads','2','-y',str(output/'%06d.png')],check=True,stdout=log,stderr=log)
    files = sorted(output.glob('*.png'))
    assert len(files) == len(frames), (vname,len(files),len(frames))
    df = pd.DataFrame({'video_id':video,'frame':frames,'seconds':frames/c['source_fps'],
                       'phase':phase.Phase.iloc[frames].map({p:i for i,p in enumerate(PHASES)}).to_numpy(),
                       'image':[str(p.relative_to(processed)) for p in files]})
    df[TOOLS] = tool[TOOLS].iloc[tool_indices].to_numpy()
    assert not df.isna().any().any()
    manifest.parent.mkdir(parents=True,exist_ok=True)
    tmp = manifest.with_suffix('.tmp')
    df.to_csv(tmp,index=False)
    tmp.replace(manifest)
    a = {'video_id':video,'fps':c['fps'],'storage_size':c['storage_size'],
         'video_frames':nvideo,'phase_rows':len(phase),'tool_rows':len(tool),
         'sampled_frames':len(frames),'first_source_frame':int(frames[0]),'last_source_frame':int(frames[-1]),
         'excluded_terminal_video_frames':max(0,nvideo-len(phase)),
         'excluded_terminal_phase_rows':max(0,len(phase)-nvideo),
         'tool_policy':'last observed tool vector; include official terminal frame at age 25',
         'terminal_tool_hold_frames':int((tool_age==25).sum()),
         'manifest_sha256':digest(manifest),'source':probe,'completed_at':now()}
    atomic_json(audit,a)
    print(f'prepared {vname}: {len(frames)} frames; source={nvideo}, phase={len(phase)}',flush=True)
    return a


def normalization(c):
    from .common import phase_weights
    processed = Path(c['processed_root'])
    stats_path=processed/'training_statistics.json'
    inputs_path=processed/'training_statistics.inputs.json'
    signature={'train_videos':ids(c,'train'),'input_size':c['input_size'],
               'storage_size':c['storage_size'],'fps':c['fps'],
               'method':'RGB_224_resize_linear_train_1hz_mean_std_v1',
               'manifests':{str(v):digest(processed/'manifests'/f'video{v:02d}.csv') for v in ids(c,'train')}}
    if stats_path.exists() and inputs_path.exists():
        assert json.loads(inputs_path.read_text())==signature, 'Training manifests changed; use a new processed/run directory'
        return json.loads(stats_path.read_text())
    frames = pd.concat([pd.read_csv(processed/'manifests'/f'video{v:02d}.csv') for v in ids(c,'train')],ignore_index=True)
    weights = phase_weights(frames.phase.to_numpy())
    # Deterministic 1 Hz sample from training patients only. No val/test statistics.
    sample = frames[frames.frame % 25 == 0]
    def sums(path):
        im = cv2.cvtColor(cv2.imread(str(processed/path)),cv2.COLOR_BGR2RGB)
        im = cv2.resize(im,(c['input_size'],c['input_size']),interpolation=cv2.INTER_LINEAR).astype(np.float64)/255
        return im.sum((0,1)), (im*im).sum((0,1)), im.shape[0]*im.shape[1]
    total = np.zeros(3); total2 = np.zeros(3); count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for a,b,n in pool.map(sums,sample.image):
            total+=a;total2+=b;count+=n
    mean = total/count
    result={'mean':mean.tolist(),'std':np.sqrt(total2/count-mean**2).tolist(),
            'class_weights':weights.tolist(),'class_counts':np.bincount(frames.phase,minlength=7).tolist(),
            'train_videos':ids(c,'train'),'normalization_sampling_fps':1,'normalization_frames':len(sample),
            'weight_formula':'median(training frame counts)/training phase count','created_at':now()}
    if stats_path.exists():
        # Adopt an earlier snapshot only after proving its numeric content agrees.
        old=json.loads(stats_path.read_text())
        assert {k:v for k,v in old.items() if k!='created_at'}=={k:v for k,v in result.items() if k!='created_at'}, 'Training statistics changed; do not invalidate existing encoders'
        result=old
    else:
        atomic_json(stats_path,result)
    atomic_json(inputs_path,signature)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/reproduction/cholec80.yaml')
    ap.add_argument('--unpack',action='store_true');ap.add_argument('--wait-download',action='store_true')
    ap.add_argument('--videos',type=int,nargs='+');ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--statistics',action='store_true');args=ap.parse_args();c=config(args.config)
    if args.unpack: unpack(c,args.wait_download)
    videos=args.videos or list(range(1,81))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(make_video,[c]*len(videos),videos))
    if args.statistics: print(normalization(c))


if __name__=='__main__': main()
