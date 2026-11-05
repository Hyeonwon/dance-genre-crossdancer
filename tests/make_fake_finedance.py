"""스모크 테스트용 가짜 FineDance (motion/*.npy (T,315) + label_json/*.json)."""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/fake_finedance")
args = ap.parse_args()
out = Path(args.out)
(out / "motion").mkdir(parents=True, exist_ok=True)
(out / "label_json").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
styles = ["Popping", "Breaking", "Locking", "Hiphop", "Jazz", "Korean"]
for i in range(12):
    T = 400
    t = np.linspace(0, 8 * np.pi, T)[:, None]
    ang = 0.3 * np.sin(t * rng.uniform(0.5, 2, (1, 52)) + rng.uniform(0, 6, (1, 52)))
    c, s = np.cos(ang), np.sin(ang)
    z = np.zeros_like(c)
    # x축 회전 행렬의 앞 두 행: [1,0,0], [0,c,-s]
    d6 = np.stack([np.ones_like(c), z, z, z, c, -s], -1).reshape(T, 52 * 6)
    trans = np.cumsum(rng.normal(0, 0.005, (T, 3)), 0)
    np.save(out / "motion" / f"{i:03d}.npy", np.concatenate([trans, d6], 1).astype(np.float32))
    (out / "label_json" / f"{i:03d}.json").write_text(json.dumps(
        {"name": f"song{i}", "style1": "Street", "style2": styles[i % len(styles)], "frames": T}))
print(f"가짜 FineDance 저장: {out}")
