"""외부 데이터셋(FineDance) 평가.

모델은 AIST++ 10개 장르 중 하나를 고른다. FineDance 장르마다 정답으로 인정하는 AIST++ 장르가 있다
(Hiphop → Middle 또는 LA Hip-hop). 두 가지 방식으로 잰다.
- open:       10개 장르 전체에서 고른 결과 (겹치지 않는 장르로 틀릴 수도 있음 → 더 엄격)
- restricted: 겹치는 장르(정답 후보) 중에서만 고른 결과 (장르 구분 능력만 봄)
"""
import numpy as np
from sklearn.metrics import f1_score

from dgc.data.finedance import FD_LABELS, GENRE_MAP
from dgc.eval.metrics import seq_aggregate


def _to_fd(code: str) -> str:
    for lab, codes in GENRE_MAP.items():
        if code in codes:
            return lab
    return "other"


def evaluate_external(win_probs: np.ndarray, win_seq: list, fd_cache: dict, g2i: dict) -> dict:
    i2g = {i: g for g, i in g2i.items()}
    names, P = seq_aggregate(win_probs, win_seq)
    true = [fd_cache["seqs"][n]["fd_label"] for n in names]
    allowed = sorted({g2i[c] for codes in GENRE_MAP.values() for c in codes if c in g2i})
    mask = np.full(P.shape[1], -np.inf)
    mask[allowed] = 0.0
    out = {"n_seqs": len(names), "n_windows": len(win_seq)}
    for mode, logits in [("open", P), ("restricted", P + mask)]:
        pred_code = [i2g[int(i)] for i in logits.argmax(1)]
        pred = [_to_fd(c) for c in pred_code]
        acc = float(np.mean([p == t for p, t in zip(pred, true)]))
        f1 = f1_score(true, pred, labels=FD_LABELS, average="macro", zero_division=0)
        per = f1_score(true, pred, labels=FD_LABELS, average=None, zero_division=0)
        out[mode] = {"seq_acc": acc, "seq_macro_f1": float(f1),
                     "per_genre_f1": dict(zip(FD_LABELS, per.tolist()))}
    # 장르별로 어떤 AIST++ 장르로 예측했는지 (open 기준) — 분석용
    conf = {lab: {} for lab in FD_LABELS}
    for t, c in zip(true, [i2g[int(i)] for i in P.argmax(1)]):
        conf[t][c] = conf[t].get(c, 0) + 1
    out["pred_counts_open"] = conf
    out["n_per_genre"] = {lab: true.count(lab) for lab in FD_LABELS}
    return out
