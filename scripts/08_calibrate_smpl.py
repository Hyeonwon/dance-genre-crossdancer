"""SMPL 관절 → COCO 17관절 보정값 추정 (FineDance 외부 검증 준비).

AIST++에는 같은 춤에 대해 (1) 실제 COCO 3D 좌표(keypoints3d)와 (2) SMPL 회전값(motions)이 둘 다 있다.
SMPL 회전값을 우리 FK로 COCO 근사 좌표로 바꾼 뒤 실제 좌표와 비교해서, 관절마다 '붙은 뼈 기준
고정 보정 벡터'를 추정한다. 같은 보정을 FineDance(SMPL-H)에 적용하면 AIST++ 학습 모델이 보던 관절
정의에 가까워진다.

시퀀스를 절반으로 나눠 앞 절반으로 추정하고, 뒤 절반으로 보정 전/후 오차를 잰다.

사용법:
    python scripts/08_calibrate_smpl.py      # → configs/smpl_calib.json, results/smpl_calib.md
"""
import json
import pickle

import _bootstrap  # noqa: F401
import numpy as np

from dgc.config import load_config, resolve
from dgc.data.aist import fill_nans, load_ignore_list
from dgc.data.smpl_fk import ATTACH, smpl_axis_angle_fk, to_coco17

NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow",
         "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]


def load_pair(anno, name):
    with open(anno / "motions" / f"{name}.pkl", "rb") as f:
        m = pickle.load(f)
    with open(anno / "keypoints3d" / f"{name}.pkl", "rb") as f:
        k = pickle.load(f)["keypoints3d"]
    s = float(np.asarray(m["smpl_scaling"]).ravel()[0])
    P, G = smpl_axis_angle_fk(m["smpl_poses"], m["smpl_trans"] / s)
    K = fill_nans(np.asarray(k, dtype=np.float64)) / s          # cm → SMPL m 단위
    T = min(len(P), len(K))
    step = 4                                                    # 60fps → 15fps 로 줄여서 빠르게
    return P[:T:step], G[:T:step], K[:T:step]


def centered(J):
    return J - 0.5 * (J[:, 11:12] + J[:, 12:13])


def residual_local(P, G, K, calib):
    """붙은 관절 좌표계로 표현한 (실제 − 근사) 차이 (T, 17, 3)."""
    J = to_coco17(P, G, calib).astype(np.float64)
    e = centered(K) - centered(J)
    return np.einsum("tkji,tkj->tki", G[:, ATTACH], e)          # Gᵀ e


def errors(P, G, K, calib):
    J = to_coco17(P, G, calib).astype(np.float64)
    return np.linalg.norm(centered(K) - centered(J), axis=-1)   # (T, 17)


def main():
    cfg = load_config()
    anno = resolve(cfg["paths"]["anno_dir"])
    if not (anno / "motions").exists():
        raise FileNotFoundError(f"{anno / 'motions'} 가 없습니다. AIST++ 릴리스의 motions.zip 을 풀어 주세요.")
    ignore = load_ignore_list(anno)
    names = sorted(p.stem for p in (anno / "motions").glob("*.pkl") if p.stem not in ignore
                   and (anno / "keypoints3d" / f"{p.stem}.pkl").exists())
    fit, held = names[0::2], names[1::2]
    print(f"짝지어진 시퀀스 {len(names)}개 (추정 {len(fit)}, 평가 {len(held)})")
    pairs_fit = [load_pair(anno, n) for n in fit]

    calib = np.zeros((17, 3))
    for it in range(3):          # 엉덩이 보정이 골반 중심을 조금 옮기므로 몇 번 반복
        r = np.concatenate([residual_local(P, G, K, calib) for P, G, K in pairs_fit])
        calib = calib + np.median(r, axis=0)
    del pairs_fit

    before, after = [], []
    for n in held:
        P, G, K = load_pair(anno, n)
        before.append(errors(P, G, K, None))
        after.append(errors(P, G, K, calib))
    b, a = np.median(np.concatenate(before), 0) * 100, np.median(np.concatenate(after), 0) * 100

    out = resolve("configs/smpl_calib.json")
    out.write_text(json.dumps({"calib_m": calib.round(5).tolist(), "joints": NAMES,
                               "n_fit": len(fit), "n_eval": len(held)}, indent=1), encoding="utf-8")
    lines = ["# SMPL → COCO 보정 결과", "",
             f"AIST++ 짝 시퀀스 {len(names)}개 중 절반으로 추정, 나머지 절반으로 평가. "
             "오차 = 골반 중심을 맞춘 뒤 관절 위치 차이의 중앙값 (cm, SMPL 평균 체형 기준).", "",
             "| 관절 | 보정 전 | 보정 후 |", "|---|---|---|"]
    lines += [f"| {NAMES[j]} | {b[j]:.1f} | {a[j]:.1f} |" for j in range(17)]
    lines += [f"| **평균** | **{b.mean():.1f}** | **{a.mean():.1f}** |"]
    resolve("results/smpl_calib.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
