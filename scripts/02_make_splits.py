"""[1단계] 분할 파일 생성 → data/splits/cross.json, seen.json
모든 모델이 같은 분할을 쓰도록 한 번만 만든다. 같은 데이터·같은 seed면 누가 돌려도 같은 결과가 나온다.
실행: python scripts/02_make_splits.py
"""
import argparse

import _bootstrap  # noqa: F401
import numpy as np

from dgc.config import load_config, resolve
from dgc.data.preprocess import load_cache
from dgc.data.splits import check_genre_dancer, make_cross_dancer, make_seen_dancer, save_splits

ap = argparse.ArgumentParser()
ap.add_argument("--set", nargs="*", default=[])
args = ap.parse_args()
cfg = load_config(overrides=args.set)
cache = load_cache(cfg)
seqs = cache["seqs"]
s = cfg["split"]

table, multi = check_genre_dancer(seqs)
for g, ds in table.items():
    print(f"  {g}: {ds}")
if multi:
    print(f"[경고] 여러 장르를 춘 댄서가 있습니다: {multi}. 댄서 기준 분할의 해석에 주의하세요.")

cross = make_cross_dancer(seqs, s["n_repeats"], s["val_frac"], s["seed"])
test_frac = float(np.mean([len(c["test"]) / len(seqs) for c in cross]))
seen = make_seen_dancer(seqs, s["n_repeats"], test_frac, s["val_frac"], s["seed"])
d = resolve(cfg["paths"]["splits_dir"])
save_splits(cross, d / "cross.json")
save_splits(seen, d / "seen.json")
print(f"cross: {len(cross)}회, 테스트 비율 평균 {test_frac:.3f}")
print(f"  rep0 테스트 댄서: {cross[0]['test_dancers']}")
print(f"  rep0 train/val/test 시퀀스: {len(cross[0]['train'])}/{len(cross[0]['val'])}/{len(cross[0]['test'])}")
print(f"seen:  {len(seen)}회, rep0 train/val/test: {len(seen[0]['train'])}/{len(seen[0]['val'])}/{len(seen[0]['test'])}")
print(f"저장: {d}")
