"""[2단계] XGBoost 기준선.
실행 예:
  python scripts/03_train_xgb.py --setting cross            # 10회 전부
  python scripts/03_train_xgb.py --setting seen --repeats 0,1,2
  python scripts/03_train_xgb.py --setting cross --set data.drop_joint_groups=[legs] --tag drop_legs
"""
import argparse
import time

import _bootstrap  # noqa: F401

from dgc.config import load_config, resolve
from dgc.data.preprocess import load_cache
from dgc.data.splits import load_splits
from dgc.data.windows import handcrafted_for_all
from dgc.train.common import parse_repeats, prep_settings, prep_version, result_path, save_result
from dgc.train.xgb import run_xgb

ap = argparse.ArgumentParser()
ap.add_argument("--setting", choices=["cross", "seen"], required=True)
ap.add_argument("--repeats", default=None, help="예: 0,1,2 (생략 시 전부)")
ap.add_argument("--tag", default="", help="결과 폴더 이름 뒤에 붙일 태그 (부위 제거 실험 등)")
ap.add_argument("--set", nargs="*", default=[])
args = ap.parse_args()

cfg = load_config(overrides=args.set)
exp = "xgb" + (f"_{args.tag}" if args.tag else "")
cache = load_cache(cfg)
splits = load_splits(resolve(cfg["paths"]["splits_dir"]) / f"{args.setting}.json", cfg, cache)
t0 = time.time()
feats = handcrafted_for_all(cache, cfg)
print(f"수작업 특징 계산 완료 ({time.time() - t0:.0f}s)")
for r in parse_repeats(args.repeats, len(splits)):
    res = run_xgb(cfg, cache, feats, splits[r], seed=r)
    res.update({"exp": exp, "setting": args.setting, "repeat": r, "config": cfg,
                    "prep_version": prep_version(cfg), "prep_settings": prep_settings(cfg)})
    p = result_path(cfg, exp, args.setting, r)
    save_result(p, res)
    print(f"[{exp} {args.setting} rep{r}] seq Macro-F1 {res['test']['seq_macro_f1']:.4f} "
          f"(depth {res['best_max_depth']}) → {p}")
