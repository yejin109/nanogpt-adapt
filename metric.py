import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


def get_token_acc(_logits, _labels):
    preds = np.argmax(_logits, -1)

    preds = preds.reshape(-1)
    _labels = _labels.reshape(-1)

    acc_token = accuracy_score(_labels, preds)

    return acc_token


def get_entropy(_logits, _labels):
    log_probs = np.log(np.exp(_logits) / (np.exp(_logits).sum(axis=-1, keepdims=True)+ 1e-6))
    entropy = - np.sum(np.exp(log_probs) * log_probs, axis=-1)
    entropy = entropy.mean(axis=-1)
    return entropy


def get_perr(_logits, _labels):
    preds = np.argmax(_logits, -1)
    p_err = np.array(list(map(lambda p, l: ~ (p == l).all(), preds, _labels))).mean()
    return p_err


def get_memorization_clm(_logits: np.ndarray, _labels: np.ndarray, mask_num):
    _preds = _logits.argmax(-1)
    return np.mean(_preds[len(_logits)-1-mask_num:] == _labels[len(_logits)-1-mask_num:])