"""XGBoost용 수작업 특징 (윈도우 1개 → 특징 벡터 1개).

관절 17개 기준 약 93개:
- 관절별 속도 크기의 평균/표준편차/최댓값, 가속도 크기 평균, 저크 크기 평균 (17×5 = 85)
- 골반 높이 평균/범위, 수평 이동 속도 평균 (3)
- 상체 대 하체 운동 에너지 비율 (1), 좌우 대칭도 (1)
- 전신 속도 신호의 FFT 주 주파수(리듬)와 그 에너지 비중 (2), 동작 공간 크기 (1)
"""
import numpy as np

from dgc.data.skeleton import LOWER, MIRROR_PAIRS, UPPER

JOINT_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
               "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
               "l_knee", "r_knee", "l_ankle", "r_ankle"]


def feature_names(keep: list[int]) -> list[str]:
    names = []
    for stat in ["speed_mean", "speed_std", "speed_max", "acc_mean", "jerk_mean"]:
        names += [f"{JOINT_NAMES[j]}_{stat}" for j in keep]
    names += ["root_height_mean", "root_height_range", "hspeed_mean"]
    names += ["upper_lower_energy_ratio", "lr_symmetry", "rhythm_freq_hz", "rhythm_power_frac",
              "motion_volume"]
    return names


def window_handcrafted(pos, vel, acc, root_h, hspeed, keep: list[int], fps: float) -> np.ndarray:
    speed = np.linalg.norm(vel, axis=-1)                  # (L, 17)
    accm = np.linalg.norm(acc, axis=-1)
    jerk = np.linalg.norm(np.diff(acc, axis=0) * fps, axis=-1)
    k = keep
    feats = [speed[:, k].mean(0), speed[:, k].std(0), speed[:, k].max(0),
             accm[:, k].mean(0), jerk[:, k].mean(0)]

    up_k = [j for j in UPPER if j in k]
    lo_k = [j for j in LOWER if j in k]
    if up_k and lo_k:
        ratio = (speed[:, up_k] ** 2).mean() / ((speed[:, lo_k] ** 2).mean() + 1e-6)
    else:
        ratio = 0.0
    sym = []
    for a, b in MIRROR_PAIRS:
        if a in k and b in k and speed[:, a].std() > 1e-6 and speed[:, b].std() > 1e-6:
            sym.append(np.corrcoef(speed[:, a], speed[:, b])[0, 1])
    sym = float(np.mean(sym)) if sym else 0.0

    s = speed[:, k].mean(1)
    s = s - s.mean()
    spec = np.abs(np.fft.rfft(s)) ** 2
    freqs = np.fft.rfftfreq(len(s), d=1.0 / fps)
    if len(spec) > 1 and spec[1:].sum() > 0:
        i = 1 + int(np.argmax(spec[1:]))
        rhythm_f, rhythm_p = freqs[i], spec[i] / spec[1:].sum()
    else:
        rhythm_f, rhythm_p = 0.0, 0.0
    rng = pos[:, k].reshape(-1, 3)
    volume = float(np.prod(rng.max(0) - rng.min(0)))

    extra = [root_h.mean(), root_h.max() - root_h.min(), hspeed.mean(),
             ratio, sym, rhythm_f, rhythm_p, volume]
    return np.concatenate([np.concatenate(feats), np.asarray(extra, dtype=np.float64)]).astype(np.float32)
