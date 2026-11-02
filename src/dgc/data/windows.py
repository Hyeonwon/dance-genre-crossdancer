"""시퀀스 → 윈도우. XGBoost(수작업 특징)와 신경망(PyTorch Dataset)이 같은 윈도우 정의를 공유한다."""
import numpy as np

from dgc.config import window_len, window_stride
from dgc.data.augment import bone_scale
from dgc.data.naming import GENRES
from dgc.data.preprocess import (frame_features, mirror, normalize_window, resample, torso_length,
                                 window_starts)
from dgc.data.skeleton import L_ANKLE, R_ANKLE, keep_joints
from dgc.features.handcrafted import window_handcrafted


def genre_index(cache) -> dict:
    present = sorted({s["genre"] for s in cache["seqs"].values()}, key=GENRES.index)
    return {g: i for i, g in enumerate(present)}


def make_window(cache, name, start, cfg, rng=None, augment=False):
    """윈도우 하나를 정규화해서 (pos, vel, acc, root_h, hspeed)로 반환."""
    s = cache["seqs"][name]
    J = s["joints"]
    L = window_len(cfg)
    a = cfg["aug"]
    if augment and a["time_scale"] > 0:
        seg = int(round(L * rng.uniform(1 - a["time_scale"], 1 + a["time_scale"])))
        seg = min(seg, J.shape[0])
        start = min(start, J.shape[0] - seg)
        W = resample(J[start:start + seg], L)
    else:
        W = J[start:start + L]
    torso = cache["torso"][s["dancer"]]
    if augment and a["bone_scale"] > 0:
        up = cache["up_axis"]
        W0 = W
        W = bone_scale(W0, rng, a["bone_scale"])
        # 가상 댄서도 '자기' 몸통 길이로 정규화되도록 몸통 길이를 같은 비율로 바꾼다
        torso = torso * torso_length(W) / torso_length(W0)
        # 다리 길이가 바뀌어도 발이 바닥에 붙어 있도록 수직 위치 보정
        ank = [L_ANKLE, R_ANKLE]
        W = W.copy()
        W[..., up] += W0[:, ank, up].min() - W[:, ank, up].min()
    pos, vel, acc, rh, hs = normalize_window(W, s["floor"], torso, cache["up_axis"], cache["fps"])
    if augment and a["mirror"] and rng.random() < 0.5:
        pos, vel, acc = mirror(pos, vel, acc)
    return pos, vel, acc, rh, hs


def index_windows(cache, names, cfg):
    L, stride = window_len(cfg), window_stride(cfg)
    items = []
    for n in names:
        for st in window_starts(cache["seqs"][n]["joints"].shape[0], L, stride):
            items.append((n, st))
    return items


def handcrafted_for_all(cache, cfg) -> dict:
    """모든 시퀀스의 윈도우 특징을 한 번에 계산: {seq: (n_windows, D)}."""
    keep = keep_joints(cfg["data"]["drop_joint_groups"])
    out = {}
    for n in cache["seqs"]:
        rows = [window_handcrafted(*make_window(cache, n, st, cfg), keep, cache["fps"])
                for _, st in index_windows(cache, [n], cfg)]
        if rows:
            out[n] = np.stack(rows)
    return out


def frame_dim(cfg) -> int:
    return 9 * len(keep_joints(cfg["data"]["drop_joint_groups"])) + 2


def window_input(cache, name, start, cfg, rng=None, augment=False) -> np.ndarray:
    keep = keep_joints(cfg["data"]["drop_joint_groups"])
    return frame_features(*make_window(cache, name, start, cfg, rng, augment), keep)
