from types import SimpleNamespace
import numpy as np
import torch
from models.mstcn import MultiStageModel
from reproduction.metrics import confusion,summarize
from reproduction.common import phase_weights
from reproduction.lightning import TeCNORuntime


def tcn(causal=True):
    torch.manual_seed(2334);torch.set_num_threads(2)
    return MultiStageModel(SimpleNamespace(mstcn_stages=2,mstcn_layers=9,mstcn_f_maps=64,
                mstcn_f_dim=2048,out_features=7,mstcn_causal_conv=causal)).eval()


def test_causal_prefix_and_future_perturbation():
    model=tcn();x=torch.randn(1,2048,87)
    changed=x.clone();changed[:,:,39:]=torch.randn_like(changed[:,:,39:])*9
    with torch.no_grad():
        original=model(x);perturbed=model(changed);prefix=model(x[:,:,:39])
    torch.testing.assert_close(original[:,:,:,:39],perturbed[:,:,:,:39],rtol=0,atol=0)
    torch.testing.assert_close(original[:,:,:,:39],prefix,atol=2e-6,rtol=2e-5)
    assert original.shape==(2,1,7,87)
    # Control: the same perturbation must affect earlier predictions without causality.
    acausal=tcn(False)
    with torch.no_grad():
        assert not torch.allclose(acausal(x)[:,:,:,:39],acausal(changed)[:,:,:,:39])


def test_future_gradient_exactly_zero():
    model=tcn();x=torch.randn(1,2048,65,requires_grad=True)
    model(x)[-1,0,2,30].backward()
    assert torch.count_nonzero(x.grad[:,:,31:])==0
    assert torch.count_nonzero(x.grad[:,:,:31])>0


def test_metrics_case_weighting_and_undefined_classes():
    # One perfect 2-frame video and one entirely incorrect 4-frame video.
    result=summarize([confusion([0,0],[0,0]),confusion([1,1,1,1],[0,0,0,0])])
    assert result['accuracy_video_mean']==.5
    assert result['accuracy_frame_pooled']==1/3
    assert result['precision_phase_then_video_macro']==.5
    assert result['recall_phase_then_video_macro']==.5
    assert result['precision_per_phase'][1] is None
    assert result['recall_per_phase'][2] is None


def test_crossentropy_uses_logits_not_probabilities():
    c={'tcn':{'mstcn_stages':2,'mstcn_layers':1,'mstcn_f_maps':4,
            'mstcn_f_dim':2048,'mstcn_causal_conv':True}}
    model=TeCNORuntime(c,'tcn',[1]*7)
    logits=torch.zeros(2,1,7,1);logits[:,:,0,:]=20
    assert model.loss_tcn(logits,torch.zeros(1,1,dtype=torch.long))<1e-6
    wrong=torch.nn.functional.cross_entropy(logits[0].softmax(1),torch.zeros(1,1,dtype=torch.long))
    assert wrong>1


def test_weights_depend_on_training_counts():
    labels=np.repeat(np.arange(7),[1,2,3,4,5,6,8])
    np.testing.assert_allclose(phase_weights(labels),[4,2,4/3,1,.8,2/3,.5])
