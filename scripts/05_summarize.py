"""[4단계] 결과 요약: 실험 × 설정별 평균 ± 표준편차, 댄서 의존도(seen − cross).
실행: python scripts/05_summarize.py  → results/summary.md
"""
import argparse
import json
from collections import defaultdict

import _bootstrap  # noqa: F401
import numpy as np

from dgc.config import load_config, resolve

ap = argparse.ArgumentParser()
ap.add_argument("--set", nargs="*", default=[])
args = ap.parse_args()
cfg = load_config(overrides=args.set)
root = resolve(cfg["paths"]["results_dir"])
root.mkdir(parents=True, exist_ok=True)
rows = defaultdict(lambda: defaultdict(list))
for p in sorted(root.glob("*/*.json")):
    r = json.loads(p.read_text())
    key = (r["exp"], r["setting"])
    rows[key]["seq_macro_f1"].append(r["test"]["seq_macro_f1"])
    rows[key]["seq_acc"].append(r["test"]["seq_acc"])
    rows[key]["window_acc"].append(r["test"]["window_acc"])
    for sit, v in r["test"].get("seq_macro_f1_by_situation", {}).items():
        rows[key][f"f1_{sit}"].append(v)
    pr = r.get("probe", {}).get("probe_acc")
    if pr is not None:
        rows[key]["probe_acc"].append(pr)
        rows[key]["probe_chance"].append(r["probe"]["probe_chance"])


def ms(v):
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else "-"


lines = ["# 결과 요약", "",
         "| 실험 | 설정 | n | 시퀀스 Macro-F1 | 시퀀스 Acc | 윈도우 Acc | Basic F1 | Advanced F1 | 개인 정보량(probe) |",
         "|---|---|---|---|---|---|---|---|---|"]
for (exp, setting), m in sorted(rows.items()):
    probe = ms(m["probe_acc"]) + (f" (우연 {np.mean(m['probe_chance']):.2f})" if m["probe_chance"] else "")
    lines.append(f"| {exp} | {setting} | {len(m['seq_macro_f1'])} | {ms(m['seq_macro_f1'])} | {ms(m['seq_acc'])} "
                 f"| {ms(m['window_acc'])} | {ms(m['f1_sBM'])} | {ms(m['f1_sFM'])} | {probe} |")

lines += ["", "## 댄서 의존도 (seen − cross, 시퀀스 Macro-F1)", "", "| 실험 | seen | cross | 의존도 |", "|---|---|---|---|"]
for exp in sorted({e for e, _ in rows}):
    s, c = rows.get((exp, "seen"), {}).get("seq_macro_f1"), rows.get((exp, "cross"), {}).get("seq_macro_f1")
    if s and c:
        lines.append(f"| {exp} | {np.mean(s):.3f} | {np.mean(c):.3f} | **{np.mean(s) - np.mean(c):+.3f}** |")

out = "\n".join(lines)
print(out)
(root / "summary.md").write_text(out, encoding="utf-8")
print(f"\n저장: {root / 'summary.md'}")
