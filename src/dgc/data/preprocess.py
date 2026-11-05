"""전처리: 다운샘플링, 캐시 생성, 윈도우 정규화, 프레임별 입력 특징.

정규화 순서 (윈도우 단위)
1. 엉덩이 중점(골반)을 매 프레임 원점으로 이동 → 전역 이동 제거
2. 윈도우 첫 프레임의 좌→우 엉덩이 방향이 +x(옆 방향)가 되도록 수직축 기준 회전
3. 댄서별 평균 몸통 길이로 나눔 → 체형(키) 차이 제거
4. 원점 이동 전 정보는 골반 높이와 수평 이동 속도로 따로 보존

출력 좌표축은 항상 (옆, 위, 앞) 순서로 맞춘다. 원본 데이터의 축 방향과 무관하게 같은 의미가 되도록.
"""
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from dgc.config import resolve
from dgc.data.aist import list_sequences, load_ignore_list, load_keypoints3d
from dgc.data.naming import parse_seq_name
from dgc.data.skeleton import (L_ANKLE, L_HIP, L_SHOULDER, MIRROR_PAIRS, R_ANKLE, R_HIP,
                               R_SHOULDER)


def downsample(arr: np.ndarray, src_fps: int, fps: int) -> np.ndarray:
    step = src_fps // fps
    if src_fps % fps != 0:
        raise ValueError("src_fps 는 fps 의 정수배여야 합니다.")
    return arr[::step]


def pelvis(arr):
    return 0.5 * (arr[..., L_HIP, :] + arr[..., R_HIP, :])


def neck(arr):
    return 0.5 * (arr[..., L_SHOULDER, :] + arr[..., R_SHOULDER, :])


def torso_length(arr: np.ndarray) -> float:
    return float(np.linalg.norm(neck(arr) - pelvis(arr), axis=-1).mean())


def floor_height(arr: np.ndarray, up: int) -> float:
    """바닥 높이 = 두 발목 높이의 하위 1% 값 (튀는 값에 강하게)."""
    ankles = np.minimum(arr[:, L_ANKLE, up], arr[:, R_ANKLE, up])
    return float(np.percentile(ankles, 1))


def build_cache(cfg) -> dict:
    """모든 시퀀스를 읽어 30fps로 줄이고, 메타데이터와 함께 하나의 pickle로 저장."""
    anno_dir = resolve(cfg["paths"]["anno_dir"])
    d = cfg["data"]
    ignore = load_ignore_list(anno_dir)
    names = [n for n in list_sequences(anno_dir) if n not in ignore]
    seqs = {}
    for i, name in enumerate(names):
        meta = parse_seq_name(name)
        arr = downsample(load_keypoints3d(anno_dir, name, d["keypoints_key"]), d["src_fps"], d["fps"])
        if d.get("up_sign", 1) == -1:          # 위쪽이 +가 되도록 뒤집기
            arr = arr.copy()
            arr[..., d["up_axis"]] *= -1
        seqs[name] = {**meta, "joints": arr.astype(np.float32), "floor": floor_height(arr, d["up_axis"])}
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(names)} 시퀀스 처리")
    # 댄서별 평균 몸통 길이 (그 댄서의 모든 시퀀스 기준)
    per_dancer = defaultdict(list)
    for s in seqs.values():
        per_dancer[s["dancer"]].append(torso_length(s["joints"]))
    torso = {k: float(np.mean(v)) for k, v in per_dancer.items()}
    cache = {"seqs": seqs, "torso": torso, "fps": d["fps"], "up_axis": d["up_axis"],
             "n_ignored": len(ignore), "built_with": {k: d.get(k) for k in CACHE_KEYS}}
    out = resolve(cfg["paths"]["cache"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(cache, f)
    print(f"캐시 저장: {out} ({len(seqs)}개 시퀀스, 제외 {len(ignore)}개)")
    return cache


# 캐시를 만들 때 쓰이는 설정. 이 중 하나라도 바뀌면 캐시를 다시 만들어야 한다.
# (window_sec, stride_sec, drop_joint_groups 는 학습 때 적용되므로 캐시를 다시 만들 필요 없음)
CACHE_KEYS = ["keypoints_key", "src_fps", "fps", "up_axis", "up_sign"]


def load_cache(cfg) -> dict:
    p = resolve(cfg["paths"]["cache"])
    if not p.exists():
        raise FileNotFoundError(f"{p} 가 없습니다. 먼저 scripts/01_build_cache.py 를 실행하세요.")
    with open(p, "rb") as f:
        cache = pickle.load(f)
    built = cache.get("built_with", {})
    now = {k: cfg["data"].get(k) for k in CACHE_KEYS}
    diff = {k: (built.get(k), now[k]) for k in CACHE_KEYS if built.get(k) != now[k]}
    if diff:
        raise RuntimeError("캐시를 만들 때와 지금 설정이 다릅니다 (캐시 값, 현재 값): "
                           f"{diff}\n→ python scripts/01_build_cache.py 와 02_make_splits.py 를 다시 실행하세요.")
    return cache


def gaussian_smooth(X: np.ndarray, sigma: float) -> np.ndarray:
    """시간축(0번 축) 가우시안 평활화. 끝은 첫/마지막 값으로 채운다 (scipy 없이)."""
    if not sigma or sigma <= 0:
        return X
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    Xp = np.concatenate([np.repeat(X[:1], r, 0), X, np.repeat(X[-1:], r, 0)])
    out = np.zeros_like(X, dtype=np.float64)
    for i, w in enumerate(k):
        out += w * Xp[i:i + len(X)]
    return out.astype(X.dtype)


def smooth_cache(cache: dict, sigma: float) -> dict:
    """캐시의 모든 시퀀스를 평활화한 복사본 (바닥 높이·몸통 길이도 다시 계산).
    AIST++ keypoints3d(여러 카메라 2D 검출을 삼각측량)는 떨림이 있어 속도·가속도가 부풀려져 있다.
    FineDance(모션캡처 기반 SMPL)와 비교할 때 이 측정 잡음 차이를 없애기 위해 쓴다."""
    if not sigma or sigma <= 0:
        return cache
    up = cache["up_axis"]
    seqs = {}
    for n, s in cache["seqs"].items():
        J = gaussian_smooth(s["joints"], sigma)
        seqs[n] = {**s, "joints": J, "floor": floor_height(J, up)}
    per = defaultdict(list)
    for s in seqs.values():
        per[s["dancer"]].append(torso_length(s["joints"]))
    return {**cache, "seqs": seqs, "torso": {k: float(np.mean(v)) for k, v in per.items()},
            "smooth_sigma": sigma}


def window_starts(T: int, L: int, stride: int) -> list[int]:
    if T < L:
        return []
    return list(range(0, T - L + 1, stride))


def _axes(up: int):
    horiz = [a for a in range(3) if a != up]
    return horiz[0], up, horiz[1]


def normalize_window(W: np.ndarray, floor: float, torso: float, up: int, fps: float):
    """W: (L, 17, 3) 원본 좌표. 반환: pos/vel/acc (L,17,3), root_h (L,), hspeed (L,)."""
    h0, u, h1 = _axes(up)
    root = pelvis(W)                                        # (L, 3)
    root_h = (root[:, u] - floor) / torso
    hvel = np.diff(root[:, [h0, h1]], axis=0, prepend=root[:1, [h0, h1]]) * fps
    hspeed = np.linalg.norm(hvel, axis=-1) / torso

    P = W - root[:, None, :]
    v = W[0, R_HIP] - W[0, L_HIP]
    ang = np.arctan2(v[h1], v[h0])
    c, s = np.cos(ang), np.sin(ang)
    x, y = P[..., h0], P[..., h1]
    lateral = c * x + s * y
    forward = -s * x + c * y
    pos = np.stack([lateral, P[..., u], forward], axis=-1) / torso   # (옆, 위, 앞)

    vel = np.diff(pos, axis=0, prepend=pos[:1]) * fps
    acc = np.diff(vel, axis=0, prepend=vel[:1]) * fps
    return pos.astype(np.float32), vel.astype(np.float32), acc.astype(np.float32), \
        root_h.astype(np.float32), hspeed.astype(np.float32)


def mirror(pos, vel, acc):
    """좌우 반전: 좌우 관절 교환 + 옆 방향 축 부호 반전."""
    idx = list(range(pos.shape[1]))
    for a, b in MIRROR_PAIRS:
        idx[a], idx[b] = b, a
    out = []
    for X in (pos, vel, acc):
        X = X[:, idx].copy()
        X[..., 0] *= -1
        out.append(X)
    return out


def frame_features(pos, vel, acc, root_h, hspeed, keep) -> np.ndarray:
    """신경망 입력: 프레임마다 [위치, 속도, 가속도](관절×3×3) + 골반 높이 + 수평 속도."""
    L = pos.shape[0]
    parts = [pos[:, keep].reshape(L, -1), vel[:, keep].reshape(L, -1), acc[:, keep].reshape(L, -1),
             root_h[:, None], hspeed[:, None]]
    return np.concatenate(parts, axis=1).astype(np.float32)


def resample(W: np.ndarray, L: int) -> np.ndarray:
    """(T, J, 3)를 시간축 선형 보간으로 길이 L로 맞춘다 (시간 스케일 증강용)."""
    T = W.shape[0]
    if T == L:
        return W
    src = np.linspace(0, T - 1, L)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (src - i0)[:, None, None]
    return (1 - w) * W[i0] + w * W[i1]
