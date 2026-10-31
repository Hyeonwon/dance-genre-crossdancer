"""[3단계] Transformer 계열 학습.
실험 이름(--exp)은 configs/exp/*.yaml:
  tf_plain    일반 Transformer (꼼수를 보여주는 기준)
  tf_aug      + 증강
  tf_adv      + 장르 조건부 적대 학습
  tf_aug_adv  + 증강 + 적대 학습 (제안 방법)
실행 예:
  python scripts/04_train_nn.py --exp tf_plain --setting cross --repeats 0,1,2 --seeds 0
"""
import argparse

import _bootstrap  # noqa: F401

from dgc.config import load_config, resolve
from dgc.data.preprocess import load_cache
from dgc.data.splits import load_splits
from dgc.train.common import parse_repeats, prep_settings, prep_version, result_path, save_result
from dgc.train.nn import run_nn

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--setting", choices=["cross", "seen"], required=True)
ap.add_argument("--repeats", default=None)
ap.add_argument("--seeds", default=None, help="예: 0,1 (생략 시 설정의 nn.seeds)")
ap.add_argument("--tag", default="")
ap.add_argument("--set", nargs="*", default=[])
args = ap.parse_args()

cfg = load_config(args.exp, args.set)
exp = args.exp + (f"_{args.tag}" if args.tag else "")
cfg["exp_name"] = exp
cache = load_cache(cfg)
splits = load_splits(resolve(cfg["paths"]["splits_dir"]) / f"{args.setting}.json", cfg, cache)
seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["nn"]["seeds"]
for r in parse_repeats(args.repeats, len(splits)):
    for seed in seeds:
        print(f"=== {exp} {args.setting} rep{r} seed{seed} ===")
        res = run_nn(cfg, cache, splits[r], seed=seed)
        res.update({"exp": exp, "setting": args.setting, "repeat": r, "config": cfg,
                    "prep_version": prep_version(cfg), "prep_settings": prep_settings(cfg)})
        p = result_path(cfg, exp, args.setting, r, seed)
        save_result(p, res)
        pr = res["probe"]["probe_acc"]
        print(f"[{exp} {args.setting} rep{r} seed{seed}] seq Macro-F1 {res['test']['seq_macro_f1']:.4f}"
              + (f", probe {pr:.3f} (우연 {res['probe']['probe_chance']:.3f})" if pr is not None else "")
              + f" → {p}")
