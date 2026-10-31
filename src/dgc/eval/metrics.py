"""평가 지표. 주 지표는 시퀀스 단위 Macro-F1 (윈도우 확률 평균으로 시퀀스 예측)."""
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def seq_aggregate(win_probs: np.ndarray, win_seq: list):
    acc = defaultdict(list)
    for p, s in zip(win_probs, win_seq):
        acc[s].append(p)
    names = sorted(acc)
    return names, np.stack([np.mean(acc[n], axis=0) for n in names])


def compute_metrics(win_probs, win_y, win_seq, cache, g2i: dict) -> dict:
    n_cls = len(g2i)
    labels = list(range(n_cls))
    win_pred = win_probs.argmax(1)
    names, seq_probs = seq_aggregate(win_probs, win_seq)
    seq_y = np.array([g2i[cache["seqs"][n]["genre"]] for n in names])
    seq_pred = seq_probs.argmax(1)
    out = {
        "window_acc": float(accuracy_score(win_y, win_pred)),
        "window_macro_f1": float(f1_score(win_y, win_pred, labels=labels, average="macro", zero_division=0)),
        "seq_acc": float(accuracy_score(seq_y, seq_pred)),
        "seq_macro_f1": float(f1_score(seq_y, seq_pred, labels=labels, average="macro", zero_division=0)),
        "per_genre_f1": dict(zip(g2i, f1_score(seq_y, seq_pred, labels=labels, average=None,
                                                zero_division=0).tolist())),
        "confusion": confusion_matrix(seq_y, seq_pred, labels=labels).tolist(),
        "n_test_seqs": len(names),
    }
    # Basic(sBM) / Advanced(sFM) 따로
    by_sit = {}
    sits = np.array([cache["seqs"][n]["situation"] for n in names])
    for sit in sorted(set(sits)):
        m = sits == sit
        by_sit[sit] = float(f1_score(seq_y[m], seq_pred[m], labels=labels, average="macro", zero_division=0))
    out["seq_macro_f1_by_situation"] = by_sit
    return out
