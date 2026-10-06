"""Lightning 2 adapters; the author model definitions remain the implementation."""
from collections import defaultdict
import random
from types import SimpleNamespace

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from models.cnn import TwoHeadResNet50Model
from models.mstcn import MultiStageModel
from .metrics import confusion,summarize


class TeCNORuntime(pl.LightningModule):
    def __init__(self, c, stage, weights, initialize_pretrained=True):
        super().__init__()
        self.save_hyperparameters({'c':c,'stage':stage,'weights':list(map(float,weights))})
        self.c=c;self.stage=stage
        if stage=='cnn':
            self.model=TwoHeadResNet50Model(SimpleNamespace(out_features=7,
                pretrained=c['cnn']['pretrained'] and initialize_pretrained))
            self.log_vars=nn.Parameter(torch.zeros(2))
        elif stage=='tcn':
            self.model=MultiStageModel(SimpleNamespace(out_features=7,**c['tcn']))
        else:raise ValueError(stage)
        self.register_buffer('class_weights',torch.tensor(weights,dtype=torch.float32))

    def forward(self,x):return self.model(x)

    def loss_cnn(self,phase_logits,tool_logits,phase,tools):
        phase_loss=F.cross_entropy(phase_logits,phase,weight=self.class_weights)
        tool_loss=F.binary_cross_entropy_with_logits(tool_logits,tools.float())
        return (torch.exp(-self.log_vars[0])*phase_loss+self.log_vars[0]
                +torch.exp(-self.log_vars[1])*tool_loss+self.log_vars[1])

    def loss_tcn(self,logits,labels):
        # CE consumes logits, with exactly one internal log_softmax.
        # Flatten time into the sample dimension: identical weighted CE, while
        # avoiding CUDA's nondeterministic spatial nll_loss2d implementation.
        return torch.stack([F.cross_entropy(s.transpose(1,2).reshape(-1,7),labels.reshape(-1),
                            weight=self.class_weights) for s in logits]).mean()

    def training_step(self,batch,batch_idx):
        if self.stage=='cnn':
            x,y,t,_,_=batch;_,p,tools=self(x);loss=self.loss_cnn(p,tools,y,t)
        else:
            x,y,_,_=batch;loss=self.loss_tcn(self(x),y)
        self.log('train_loss',loss,on_step=False,on_epoch=True,batch_size=len(y))
        return loss

    def on_validation_epoch_start(self):
        self.val_cms=defaultdict(lambda:np.zeros((7,7),dtype=np.int64))

    def validation_step(self,batch,batch_idx):
        if self.stage=='cnn':
            x,y,t,v,_=batch;_,p,tools=self(x);loss=self.loss_cnn(p,tools,y,t)
            pred=p.argmax(1).cpu().numpy();truth=y.cpu().numpy();videos=v.cpu().numpy()
            for video in np.unique(videos):
                keep=videos==video
                self.val_cms[int(video)]+=confusion(truth[keep],pred[keep])
        else:
            x,y,v,_=batch;out=self(x);loss=self.loss_tcn(out,y)
            for i,video in enumerate(v):
                self.val_cms[int(video)]+=confusion(y[i].cpu().numpy(),out[-1,i].argmax(0).cpu().numpy())
        self.log('val_loss',loss,on_epoch=True,batch_size=len(y))

    def on_validation_epoch_end(self):
        result=summarize([self.val_cms[v] for v in sorted(self.val_cms)])
        self.log('val_acc_video',result['accuracy_video_mean'],prog_bar=True)
        self.log('val_acc_frame',result['accuracy_frame_pooled'])
        self.log('val_precision',result['precision_phase_then_video_macro'])
        self.log('val_recall',result['recall_phase_then_video_macro'])
        self.latest_validation=result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.c[self.stage]['learning_rate'])

    def on_save_checkpoint(self,checkpoint):
        checkpoint['reproduction_rng']={'python':random.getstate(),'numpy':np.random.get_state(),
                'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}

    def on_load_checkpoint(self,checkpoint):
        if 'reproduction_rng' in checkpoint:
            rng=checkpoint['reproduction_rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda'])
