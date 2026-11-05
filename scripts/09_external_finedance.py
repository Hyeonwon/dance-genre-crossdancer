"""[확장] FineDance 외부 검증: AIST++ 전체(30명)로 학습 → FineDance(다른 댄서·스튜디오·장비)로 테스트.

AIST++와 겹치는 5개 장르(Break, Pop, Lock, Hip-hop, Jazz)만 평가한다. 학습에는 FineDance를 쓰지 않는다.

실행 예:
  python scripts/09_external_finedance.py --exp xgb
  python scripts/09_external_finedance.py --exp tf_plain --seeds 0,1,2
  python scripts/09_external_finedance.py --exp tf_aug_adv --seeds 0,1,2
  python scripts/09_external_finedance.py --summary          # 표만 출력
결과: results/external_finedance/<exp>_seed<k>_v<버전>.json, 요약: results/external_finedance_summary.md
"""
import argparse
import json
import time
from collections import defaultdict

import _bootstrap  # noqa: F401
import numpy as np

from dgc.config import load_config, resolve
from dgc.data.finedance import FD_LABELS, load_finedance_cache
from dgc.data.preprocess import load_cache, smooth_cache
from dgc.data.splits import _split_val
from dgc.data.windows import handcrafted_for_all
from dgc.eval.external import evaluate_external
from dgc.train.common import prep_settings, prep_version, save_result

def out_dir(cfg):
    return resolve(cfg["paths"]["results_dir"]) / "external_finedance"


def summary(cfg):
    rows = defaultdict(list)
    for f in sorted(out_dir(cfg).glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        rows[r["exp"]].append(r["external"])
    if not rows:
        print("결과가 없습니다.")
        return
    ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"
    lines = ["# FineDance 외부 검증 (AIST++ 전체로 학습)", "",
             "| 실험 | n | 10개 장르 중 선택 Macro-F1 | 정확도 | 겹치는 장르 중 선택 Macro-F1 | 정확도 |",
             "|---|---|---|---|---|---|"]
    order = ["xgb", "tf_plain", "tf_aug", "tf_adv", "tf_aug_adv"]
    for exp in sorted(rows, key=lambda e: (order.index(e) if e in order else 99, e)):
        ex = rows[exp]
        lines.append(f"| {exp} | {len(ex)} | {ms([e['open']['seq_macro_f1'] for e in ex])} | "
                     f"{ms([e['open']['seq_acc'] for e in ex])} | "
                     f"{ms([e['restricted']['seq_macro_f1'] for e in ex])} | "
                     f"{ms([e['restricted']['seq_acc'] for e in ex])} |")
    lines += ["", "장르별 F1 (10개 장르 중 선택, 평균)", "",
              "| 실험 | " + " | ".join(FD_LABELS) + " |", "|---|" + "---|" * len(FD_LABELS)]
    for exp in sorted(rows, key=lambda e: (order.index(e) if e in order else 99, e)):
        ex = rows[exp]
        lines.append(f"| {exp} | " + " | ".join(
            f"{np.mean([e['open']['per_genre_f1'][g] for e in ex]):.2f}" for g in FD_LABELS) + " |")
    n = rows[next(iter(rows))][0]["n_per_genre"]
    lines += ["", "평가 시퀀스 수: " + ", ".join(f"{g} {n[g]}" for g in FD_LABELS)]
    text = "\n".join(lines)
    (out_dir(cfg).parent / "external_finedance_summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=None, help="xgb 또는 configs/exp 의 실험 이름")
ap.add_argument("--seeds", default="0")
ap.add_argument("--tag", default="")
ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--rebuild", action="store_true", help="FineDance 캐시를 다시 만든다")
ap.add_argument("--summary", action="store_true")
args = ap.parse_args()

is_xgb = args.exp == "xgb"
if args.summary or not args.exp:
    summary(load_config(overrides=args.set))
    raise SystemExit

cfg = load_config(None if is_xgb else args.exp, args.set)
exp = args.exp + (f"_{args.tag}" if args.tag else "")
cfg["exp_name"] = exp
sigma = cfg["finedance"].get("smooth_sigma", 0)
cache = smooth_cache(load_cache(cfg), sigma)
fd = smooth_cache(load_finedance_cache(cfg, rebuild=args.rebuild), sigma)
print(f"시간축 평활화 sigma={sigma} 프레임 (AIST++·FineDance 둘 다)")

# AIST++ 전체 시퀀스로 학습 (검증용 일부만 떼어 둠). 테스트는 FineDance만.
seqs = cache["seqs"]
rng = np.random.default_rng(cfg["split"]["seed"])
train, val = _split_val(sorted(seqs), seqs, cfg["split"]["val_frac"], rng)
split = {"train": train, "val": val, "test": []}
print(f"AIST++ 학습 {len(train)} / 검증 {len(val)} 시퀀스 → FineDance {len(fd['seqs'])} 시퀀스로 테스트")

if is_xgb:
    from dgc.train.xgb import run_xgb
    t0 = time.time()
    feats = handcrafted_for_all(cache, cfg)
    fd_feats = handcrafted_for_all(fd, cfg)
    print(f"수작업 특징 계산 완료 ({time.time() - t0:.0f}s)")
    runs = [(s, lambda s=s: run_xgb(cfg, cache, feats, split, seed=s,
                                    external=(fd, fd_feats, evaluate_external))) for s in
            [int(x) for x in args.seeds.split(",")]]
else:
    from dgc.train.nn import run_nn
    runs = [(s, lambda s=s: run_nn(cfg, cache, split, seed=s, external=(fd, evaluate_external)))
            for s in [int(x) for x in args.seeds.split(",")]]

for seed, run in runs:
    print(f"=== {exp} FineDance seed{seed} ===")
    res = run()
    res.pop("history", None) if is_xgb else None
    res.update({"exp": exp, "setting": "external_finedance", "seed": seed, "config": cfg,
                "prep_version": prep_version(cfg), "prep_settings": prep_settings(cfg)})
    p = out_dir(cfg) / f"{exp}_seed{seed}_v{prep_version(cfg)}.json"
    save_result(p, res)
    e = res["external"]
    print(f"[{exp} seed{seed}] FineDance Macro-F1 {e['open']['seq_macro_f1']:.3f} "
          f"(겹치는 장르 중 선택 {e['restricted']['seq_macro_f1']:.3f}) → {p}")
summary(cfg)
