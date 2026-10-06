"""Strict frame metrics, no transition tolerance or prediction postprocessing."""
import numpy as np


def confusion(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    assert truth.shape == prediction.shape and truth.ndim == 1 and truth.size
    assert np.isin(truth, range(7)).all() and np.isin(prediction, range(7)).all()
    return np.bincount(7 * truth + prediction, minlength=49).reshape(7, 7)


def summarize(matrices):
    cms = np.asarray(matrices, dtype=np.float64)
    tp = np.diagonal(cms, axis1=1, axis2=2)
    def divide(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)
    precision = divide(tp, cms.sum(1))
    recall = divide(tp, cms.sum(2))
    acc = tp.sum(1) / cms.sum((1, 2))
    # Mean over videos with defined denominator, then mean over phases.
    def column_mean(x):
        n = np.isfinite(x).sum(0)
        return divide(np.nansum(x, axis=0), n)
    pp, rr = column_mean(precision), column_mean(recall)
    pooled = cms.sum(0)
    return {
        'accuracy_video_mean': float(acc.mean()),
        'accuracy_video_std_ddof0': float(acc.std()),
        'accuracy_frame_pooled': float(np.trace(pooled) / pooled.sum()),
        'precision_phase_then_video_macro': float(np.nanmean(pp)),
        'recall_phase_then_video_macro': float(np.nanmean(rr)),
        'precision_per_phase': [None if np.isnan(x) else float(x) for x in pp],
        'recall_per_phase': [None if np.isnan(x) else float(x) for x in rr],
        'precision_defined_videos_per_phase': np.isfinite(precision).sum(0).tolist(),
        'recall_defined_videos_per_phase': np.isfinite(recall).sum(0).tolist(),
        'accuracy_per_video': acc.tolist(),
        'confusion_pooled': pooled.astype(int).tolist(),
        'num_videos': len(cms), 'num_frames': int(cms.sum()),
        'undefined_policy': 'exclude zero-denominator video-phase pairs; report support',
    }
