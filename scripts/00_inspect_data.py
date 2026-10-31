"""[1단계 첫 작업] 데이터 구조 확인.

- 장르별 시퀀스 수, 장르별 댄서 구성 (장르당 3명이고 겹치지 않는지)
- 시퀀스 길이 분포
- 수직축(up_axis) 추정: 발목이 골반보다 일관되게 아래에 있는 축
결과는 터미널에 출력되고 results/data_report.md 에도 저장된다.

실행: python scripts/00_inspect_data.py [--limit 200]
"""
import argparse
from collections import Counter, defaultdict

import _bootstrap  # noqa: F401
import numpy as np

from dgc.config import load_config, resolve
from dgc.data.aist import list_sequences, load_ignore_list, load_keypoints3d
from dgc.data.naming import GENRE_NAMES, parse_seq_name

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=150, help="수직축 추정에 쓸 시퀀스 수")
ap.add_argument("--set", nargs="*", default=[])
args = ap.parse_args()

cfg = load_config(overrides=args.set)
anno = resolve(cfg["paths"]["anno_dir"])
ignore = load_ignore_list(anno)
names = list_sequences(anno)
metas = [parse_seq_name(n) for n in names]
used = [m for m in metas if m["name"] not in ignore]

lines = ["# AIST++ 데이터 확인 보고서", "",
         f"- keypoints3d 시퀀스: {len(names)}개, ignore_list: {len(ignore)}개, 사용: {len(used)}개", ""]

genre_d = defaultdict(Counter)
sit = Counter()
for m in used:
    genre_d[m["genre"]][m["dancer"]] += 1
    sit[(m["genre"], m["situation"])] += 1
dancer_genres = defaultdict(set)
for g, c in genre_d.items():
    for d in c:
        dancer_genres[d].add(g)

lines += ["## 장르별 댄서 구성", "", "| 장르 | 시퀀스 | 댄서 (시퀀스 수) | sBM | sFM |", "|---|---|---|---|---|"]
for g in sorted(genre_d):
    c = genre_d[g]
    ds = ", ".join(f"{d}({k})" for d, k in sorted(c.items()))
    lines.append(f"| {g} {GENRE_NAMES.get(g, '')} | {sum(c.values())} | {ds} | {sit[(g, 'sBM')]} | {sit[(g, 'sFM')]} |")
multi = {d: sorted(gs) for d, gs in dancer_genres.items() if len(gs) > 1}
n_d = [len(c) for c in genre_d.values()]
lines += ["", f"- 장르당 댄서 수: 최소 {min(n_d)}, 최대 {max(n_d)}",
          f"- 여러 장르를 춘 댄서: {multi if multi else '없음 (댄서 = 장르 1개, 계획의 전제와 일치)'}", ""]

# 길이와 수직축
rng = np.random.default_rng(0)
pick = rng.choice(len(used), min(args.limit, len(used)), replace=False)
lens, votes = [], Counter()
src_fps = cfg["data"]["src_fps"]
for i in pick:
    arr = load_keypoints3d(anno, used[i]["name"], cfg["data"]["keypoints_key"])
    lens.append(arr.shape[0] / src_fps)
    pelvis = 0.5 * (arr[:, 11] + arr[:, 12])
    ankle = 0.5 * (arr[:, 15] + arr[:, 16])
    diff = (pelvis - ankle).mean(0)          # 서 있으면 수직축에서 골반이 발목보다 크게 위
    ax = int(np.argmax(np.abs(diff)))
    votes[(ax, "+" if diff[ax] > 0 else "-")] += 1
lens = np.array(lens)
(ax, sign), cnt = votes.most_common(1)[0]
lines += ["## 시퀀스 길이 (샘플)", "",
          f"- {len(lens)}개 샘플: 평균 {lens.mean():.1f}초, 최소 {lens.min():.1f}초, 최대 {lens.max():.1f}초", "",
          "## 수직축 추정", "",
          f"- 추정: 축 {ax} ({'xyz'[ax]}), 방향 {sign} ({cnt}/{len(pick)} 시퀀스 일치)",
          f"- 현재 설정 data.up_axis = {cfg['data']['up_axis']}"
          + ("  → 일치" if ax == cfg["data"]["up_axis"] else "  → **불일치! configs/default.yaml 수정 필요**"),
          f"- 현재 설정 data.up_sign = {cfg['data'].get('up_sign', 1)}"
          + ("  → 일치" if (sign == '+') == (cfg['data'].get('up_sign', 1) == 1) else "  → **불일치! up_sign 수정 필요**")]

report = "\n".join(lines)
print(report)
out = resolve(cfg["paths"]["results_dir"]) / "data_report.md"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(report, encoding="utf-8")
print(f"\n저장: {out}")
