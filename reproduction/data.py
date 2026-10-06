from pathlib import Path
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .common import TOOLS, ids


def load_manifest(c, split=None, videos=None):
    videos = ids(c,split) if videos is None else videos
    root = Path(c['processed_root'])
    df = pd.concat([pd.read_csv(root/'manifests'/f'video{v:02d}.csv') for v in videos],ignore_index=True)
    assert set(df.video_id.unique()) == set(videos)
    assert not df.duplicated(['video_id','frame']).any()
    return df


class FrameDataset(Dataset):
    def __init__(self,c,split=None,videos=None,augment=False,limit=None):
        self.root=Path(c['processed_root'])
        self.df=load_manifest(c,split,videos)
        if limit: self.df=self.df.iloc[:limit].copy()
        stats=json.loads((self.root/'training_statistics.json').read_text())
        ops=[A.Resize(c['input_size'],c['input_size'])]
        if augment:
            ops.append(A.ShiftScaleRotate(shift_limit=.1,scale_limit=(-.2,.5),rotate_limit=15,
                                         border_mode=0,value=0,p=.7))
        ops += [A.Normalize(mean=stats['mean'],std=stats['std']),ToTensorV2()]
        self.transform=A.Compose(ops)
        self.images=self.df.image.to_numpy()
        self.phase=self.df.phase.to_numpy(np.int64)
        self.tools=self.df[TOOLS].to_numpy(np.float32)
        self.video=self.df.video_id.to_numpy(np.int64)
        self.frames=self.df.frame.to_numpy(np.int64)

    def __len__(self): return len(self.df)

    def __getitem__(self,i):
        cv2.setNumThreads(0)
        path=self.root/self.images[i]
        im=cv2.imread(str(path))
        if im is None: raise OSError(f'Cannot read {path}')
        x=self.transform(image=cv2.cvtColor(im,cv2.COLOR_BGR2RGB))['image']
        return x,int(self.phase[i]),self.tools[i],int(self.video[i]),int(self.frames[i])


class VideoDataset(Dataset):
    def __init__(self,c,split,limit=None):
        self.root=Path(c['features_root']);self.videos=ids(c,split);self.limit=limit
        self.fps=c['fps'];self.config=c
        cache=json.loads((self.root/'provenance.json').read_text())
        assert cache['fps']==self.fps and cache['dimension']==2048
        assert cache['encoder_train_videos']==ids(c,'train')
        self.encoder_hash=cache['encoder_sha256']

    def __len__(self):return len(self.videos)

    def __getitem__(self,i):
        v=self.videos[i]
        with np.load(self.root/f'video{v:02d}.npz',allow_pickle=False) as p:
            assert p['encoder_sha256'].item()==self.encoder_hash
            x=p['features'];y=p['phase'];frame=p['frame']
            assert x.shape==(len(y),2048) and np.isfinite(x).all()
            assert np.array_equal(frame,np.arange(len(frame))*(25//self.fps))
            if self.limit:x=x[:self.limit];y=y[:self.limit];frame=frame[:self.limit]
            return torch.from_numpy(x.copy()).T,torch.from_numpy(y.copy()).long(),v,frame.copy()
