import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


def get_token_acc(_logits, _labels):
    preds = np.argmax(_logits, -1)

    preds = preds.reshape(-1)
    _labels = _labels.reshape(-1)

    mask = _labels != -100
    _labels = _labels[mask]
    preds = preds[mask]

    acc_token = accuracy_score(_labels, preds)

    return acc_token


def get_entropy(_logits, _labels):
    # Entropy
    mask = _labels != -100
    log_probs = np.log(np.exp(_logits) / np.exp(_logits).sum(axis=-1, keepdims=True))
    entropy = - np.sum(np.exp(log_probs) * log_probs, axis=-1)
    entropy = entropy[mask].mean(axis=-1)
    return entropy


def get_perr(_logits, _labels):
    preds = np.argmax(_logits, -1)
    mask = _labels != -100
    p_err = np.array(list(map(lambda p, l, m: ~ (p[m] == l[m]).all(), preds, _labels, mask))).mean()
    return p_err


def get_memorization_clm(_logits: torch.Tensor, _labels: torch.Tensor, mask_num):
    def get_mask(_datum):
        return np.arange(len(_datum)-mask_num, len(_datum), dtype=int)
    _chunk_size = _logits.size(1)

    mask = list(map(lambda x: get_mask(x), _logits))
    mask_onehot = nn.functional.one_hot(torch.Tensor(mask).long().to('cuda:0'), num_classes=_chunk_size).bool()

    _preds = _logits.detach().cpu().numpy().argmax(-1)

    _labels = _labels.detach().cpu().numpy()
    mask_onehot = mask_onehot.detach().cpu().numpy()
    return np.mean(_preds[mask_onehot] == _labels[mask_onehot])