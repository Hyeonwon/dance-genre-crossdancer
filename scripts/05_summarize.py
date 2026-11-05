"""[4단계] 결과 요약: 실험 × 설정별 평균 ± 표준편차, 댄서 의존도(seen − cross).

전처리 설정이 다른 결과는 서로 비교하면 안 되므로 '전처리 버전'(prep_version)별로 따로 묶는다.
가장 최근 버전이 맨 위에 나온다. 예전 버전 결과는 지우지 않아도 섞이지 않는다.

실행: python scripts/05_summarize.py  → results/summary.md (git에 올리지 않음, 각자 생성)
"""
import argparse
import json
from collections import defaultdict

import _bootstrap  # noqa: F401
import numpy as np

from dgc.config import load_config, resolve
from dgc.train.common import prep_version

ap = argparse.ArgumentParser()
ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--all", action="store_true", help="예전 전처리 버전 결과도 모두 출력")
args = ap.parse_args()
cfg = load_config(overrides=args.set)
root = resolve(cfg["paths"]["results_dir"])
root.mkdir(parents=True, exist_ok=True)
current = prep_version(cfg)

rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # ver -> (exp, setting) -> metric
settings, newest = {}, defaultdict(float)
for p in sorted(root.glob("*/*.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    if not r.get("test"):          # 외부 검증 결과(FineDance)는 09 스크립트에서 따로 요약
        continue
    ver = r.get("prep_version", "unknown")
    settings[ver] = r.get("prep_settings", {})
    newest[ver] = max(newest[ver], p.stat().st_mtime)
    m = rows[ver][(r["exp"], r["setting"])]
    m["seq_macro_f1"].append(r["test"]["seq_macro_f1"])
    m["seq_acc"].append(r["test"]["seq_acc"])
    m["window_acc"].append(r["test"]["window_acc"])
    for sit, v in r["test"].get("seq_macro_f1_by_situation", {}).items():
        m[f"f1_{sit}"].append(v)
    pr = r.get("probe", {}).get("probe_acc")
    if pr is not None:
        m["probe_acc"].append(pr)
        m["probe_chance"].append(r["probe"]["probe_chance"])


def ms(v):
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else "-"


def section(ver, table):
    tag = " (현재 설정)" if ver == current else ""
    out = [f"## 전처리 버전 `{ver}`{tag}", "", f"설정: `{json.dumps(settings.get(ver, {}), ensure_ascii=False)}`", "",
           "| 실험 | 설정 | n | 시퀀스 Macro-F1 | 시퀀스 Acc | 윈도우 Acc | Basic F1 | Advanced F1 | 개인 정보량(probe) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for (exp, setting), m in sorted(table.items()):
        probe = ms(m["probe_acc"]) + (f" (우연 {np.mean(m['probe_chance']):.2f})" if m["probe_chance"] else "")
        out.append(f"| {exp} | {setting} | {len(m['seq_macro_f1'])} | {ms(m['seq_macro_f1'])} | {ms(m['seq_acc'])} "
                   f"| {ms(m['window_acc'])} | {ms(m['f1_sBM'])} | {ms(m['f1_sFM'])} | {probe} |")
    out += ["", "### 댄서 의존도 (seen − cross, 시퀀스 Macro-F1)", "", "| 실험 | seen | cross | 의존도 |", "|---|---|---|---|"]
    for exp in sorted({e for e, _ in table}):
        s, c = table.get((exp, "seen"), {}).get("seq_macro_f1"), table.get((exp, "cross"), {}).get("seq_macro_f1")
        if s and c:
            out.append(f"| {exp} | {np.mean(s):.3f} | {np.mean(c):.3f} | **{np.mean(s) - np.mean(c):+.3f}** |")
    return out + [""]


order = sorted(rows, key=lambda v: (v != current, -newest[v]))
lines = ["# 결과 요약", ""]
if len(order) > 1:
    lines += [f"> 전처리 설정이 다른 결과가 {len(order)}종류 있습니다. 버전별로 따로 묶었으며, 서로 비교하면 안 됩니다."
              + ("" if args.all else " 현재 설정 버전만 표시합니다 (전부 보려면 --all)."), ""]
for ver in (order if args.all else order[:1]):
    lines += section(ver, rows[ver])
if current not in rows:
    lines += [f"> 현재 설정(`{current}`)으로 돌린 결과가 아직 없습니다. 위는 가장 최근 버전입니다."]

out = "\n".join(lines)
print(out)
(root / "summary.md").write_text(out, encoding="utf-8")
print(f"\n저장: {root / 'summary.md'}")
