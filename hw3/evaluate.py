from collections import defaultdict

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from mir_eval.util import midi_to_hz
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes

from dataset import MAESTRO_small, allocate_batch
from constants import HOP_SIZE, SAMPLE_RATE, MIN_MIDI

from mir_eval.util import midi_to_hz


def evaluate(model, batch, device):
    metrics = defaultdict(list)
    batch = allocate_batch(batch, device)

    frame_logit, onset_logit = model(batch['audio'])

    criterion = nn.BCEWithLogitsLoss()
    frame_loss = criterion(frame_logit, batch['frame'])
    onset_loss = criterion(frame_logit, batch['onset'])
    metrics['metric/loss/frame_loss'].append(frame_loss.cpu().numpy())
    metrics['metric/loss/onset_loss'].append(onset_loss.cpu().numpy())


    for n in range(batch['audio'].shape[0]):
        frame_pred = th.sigmoid(frame_logit[n])
        onset_pred = th.sigmoid(onset_logit[n])

        pr, re, f1 = framewise_eval(frame_pred, batch['frame'][n])
        metrics['metric/frame/frame_precision'].append(pr)
        metrics['metric/frame/frame_recall'].append(re)
        metrics['metric/frame/frame_f1'].append(f1)

        pr, re, f1 = framewise_eval(onset_pred, batch['onset'][n])
        metrics['metric/frame/onset_precision'].append(pr)
        metrics['metric/frame/onset_recall'].append(re)
        metrics['metric/frame/onset_f1'].append(f1)

        p_est, i_est = extract_notes(onset_pred, frame_pred)
        p_ref, i_ref = extract_notes(batch['onset'][n], batch['frame'][n])

        scaling = HOP_SIZE / SAMPLE_RATE

        i_ref = (i_ref * scaling).reshape(-1, 2)
        p_ref = np.array([midi_to_hz(MIN_MIDI + pitch) for pitch in p_ref])
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + pitch) for pitch in p_est])

        p, r, f, o = evaluate_notes(
            i_ref, p_ref, i_est, p_est, offset_ratio=None)
        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        metrics['metric/note-with-offsets/precision'].append(p)
        metrics['metric/note-with-offsets/recall'].append(r)
        metrics['metric/note-with-offsets/f1'].append(f)
        metrics['metric/note-with-offsets/overlap'].append(o)

    return metrics


def extract_notes(onsets, frames, onset_threshold=0.5, frame_threshold=0.5):
    """
    Finds the note timings based on the onsets and frames information

    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float

    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    """
    onsets = (onsets > onset_threshold).type(th.int).cpu()
    frames = (frames > frame_threshold).type(th.int).cpu()
    onset_diff = th.cat(
        [onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

    pitches = []
    intervals = []

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            offset += 1
            if offset == onsets.shape[0]:
                break
            if (offset != onset) and onsets[offset, pitch].item():
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])

    return np.array(pitches), np.array(intervals)
    

def framewise_eval(pred, label, threshold=0.5):
    '''
    evaluate frame-wise (point-wise) evaluation
    pred: torch.tensor shape of (frame, pitch)
    label: torch.tensor shape of (frame, pitch)
    '''

    tp = th.sum((pred >= threshold) * (label == 1)).cpu().numpy()
    fn = th.sum((pred <  threshold) * (label == 1)).cpu().numpy()
    fp = th.sum((pred >= threshold) * (label != 1)).cpu().numpy()
    
    pr = tp / float(tp + fp) if (tp + fp) > 0 else 0
    re = tp / float(tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * pr * re / float(pr + re) if (pr + re) > 0 else 0
    
    return pr, re, f1