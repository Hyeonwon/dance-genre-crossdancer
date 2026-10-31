"""AIST++와 같은 형식의 가짜 데이터 생성 (진짜 데이터 없이 파이프라인을 점검하는 용도).

장르마다 동작 패턴(박자, 팔/다리 진폭, 들썩임)이 다르고, 댄서마다 체형과 '습관'(고유한 추가 진동)이
있어서 seen 설정이 cross 설정보다 쉽게 나오도록 만들어 둔다.
실행: python tests/make_fake_aist.py --out data/fake_aist
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

GENRES = ["gBR", "gPO", "gLO", "gWA", "gMH", "gLH", "gHO", "gKR", "gJS", "gJB"]
T0 = np.array([[0, 160, 5], [-3, 165, 3], [3, 165, 3], [-7, 162, -2], [7, 162, -2],
               [-18, 140, 0], [18, 140, 0], [-20, 112, 0], [20, 112, 0], [-20, 88, 0], [20, 88, 0],
               [-10, 95, 0], [10, 95, 0], [-10, 50, 0], [10, 50, 0], [-10, 8, 0], [10, 8, 0]], float)
ARMS, LEGS = [7, 8, 9, 10], [13, 14, 15, 16]


def make_seq(rng, g, d_params, seconds, fps=60):
    gi = GENRES.index(g)
    freq = 1.0 + 0.25 * gi
    arm_a, leg_a, bounce = 6 + 3 * (gi % 4), 4 + 3 * ((gi + 2) % 3), 2 + gi % 5
    scale, habit_f, habit_a = d_params
    t = np.arange(int(seconds * fps)) / fps
    J = np.repeat(T0[None] * scale, len(t), axis=0)
    ph = rng.uniform(0, 2 * np.pi)
    J[:, ARMS, 1] += arm_a * np.sin(2 * np.pi * freq * t + ph)[:, None]
    J[:, ARMS, 0] += arm_a * 0.5 * np.cos(2 * np.pi * freq * t)[:, None] * np.array([-1, 1, -1, 1])
    J[:, LEGS, 2] += leg_a * np.sin(2 * np.pi * freq * 0.5 * t)[:, None] * np.array([1, -1, 1, -1])
    J[:, :, 1] += bounce * np.abs(np.sin(2 * np.pi * freq * t))[:, None]
    J[:, :5, 0] += habit_a * np.sin(2 * np.pi * habit_f * t)[:, None]       # 댄서 습관(머리 흔들기)
    J[:, [9, 10], 1] += habit_a * np.cos(2 * np.pi * habit_f * t)[:, None]  # 댄서 습관(손목)
    yaw = rng.uniform(0, 2 * np.pi) + 0.3 * np.sin(0.2 * t)
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    x, z = J[..., 0].copy(), J[..., 2].copy()
    J[..., 0], J[..., 2] = c * x - s * z, s * x + c * z
    J[..., [0, 2]] += np.cumsum(rng.normal(0, 0.3, (len(t), 1, 2)), axis=0)
    J += rng.normal(0, 0.5, J.shape)
    return J.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fake_aist")
    ap.add_argument("--basic", type=int, default=4)
    ap.add_argument("--advanced", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "keypoints3d").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    names = []
    for gi, g in enumerate(GENRES):
        for k in range(3):
            d = f"d{gi * 3 + k + 1:02d}"
            params = (rng.uniform(0.9, 1.1), rng.uniform(0.3, 3.0), rng.uniform(3, 6))
            for sit, n in (("sBM", args.basic), ("sFM", args.advanced)):
                for c in range(n):
                    name = f"{g}_{sit}_cAll_{d}_m{g[1:]}{c % 6}_ch{c + 1:02d}"
                    J = make_seq(rng, g, params, rng.uniform(7.5, 14))
                    with open(out / "keypoints3d" / f"{name}.pkl", "wb") as f:
                        pickle.dump({"keypoints3d": J}, f)
                    names.append(name)
    (out / "ignore_list.txt").write_text(names[0] + "\n")
    print(f"가짜 데이터 {len(names)}개 시퀀스 → {out}")


if __name__ == "__main__":
    main()
