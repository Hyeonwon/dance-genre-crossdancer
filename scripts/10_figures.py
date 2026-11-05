"""[6단계] 발표용 그래프. results/ 의 json 만 읽어서 results/analysis/fig*.png 로 저장.

  python scripts/10_figures.py

1 fig1_performance.png   실험별 cross/seen Macro-F1 (반복 10개 평균 ± 표준편차)
2 fig2_dependency.png    댄서 의존도 (seen − cross)
3 fig3_lambda.png        적대 학습 세기별 cross Macro-F1 (시드 0)
4 fig4_probe.png         표현에 남은 댄서 정보 (probe 정확도, 우연 0.5)
5 fig5_bodypart.png      XGBoost 신체 부위 제거
6 fig6_finedance.png     FineDance 외부 검증
같은 숫자를 results/analysis/figures_numbers.md 로도 저장 (발표 대본·질의응답용).
"""
import json
from collections import defaultdict

import _bootstrap  # noqa: F401
import numpy as np

from dgc.config import resolve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})

R = resolve("results")
OUT = R / "analysis"
MODELS = [("xgb", "XGBoost"), ("tf_plain_v2", "Transformer"), ("tf_adv_v2", "+ Adv"),
          ("tf_aug_v2", "+ Aug"), ("tf_aug_adv_v2", "+ Aug + Adv\n(ours)")]


def per_rep(exp, setting, key=lambda r: r["test"]["seq_macro_f1"], seeds=None):
    """반복별로 시드 평균한 값 리스트 (반복 순서)."""
    by = defaultdict(list)
    for f in sorted((R / exp).glob(f"{setting}_rep*.json")):
        name = f.name
        if seeds is not None and not any(f"_seed{s}_" in name for s in seeds):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        by[r["repeat"]].append(key(r))
    return np.array([np.mean(by[k]) for k in sorted(by)])


def bar_ends(ax, x, h, w, color, err=None, label=None):
    ax.bar(x, h, w, color=color, edgecolor="white", linewidth=2, label=label, zorder=2)
    if err is not None:
        ax.errorbar(x, h, yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=3)


def style(ax, ylabel, ylim=None):
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color="#ecebe7", zorder=0)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    notes = ["# 발표 그래프 숫자", ""]

    # 1, 2: 성능과 댄서 의존도
    cross = {e: per_rep(e, "cross") for e, _ in MODELS}
    seen = {e: per_rep(e, "seen") for e, _ in MODELS}
    x = np.arange(len(MODELS))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.38
    bar_ends(ax, x - w / 2, [seen[e].mean() for e, _ in MODELS], w, ORANGE,
             [seen[e].std() for e, _ in MODELS], "seen dancers")
    bar_ends(ax, x + w / 2, [cross[e].mean() for e, _ in MODELS], w, BLUE,
             [cross[e].std() for e, _ in MODELS], "unseen dancers (cross)")
    for i, (e, _) in enumerate(MODELS):
        ax.text(x[i] + w / 2, cross[e].mean() + cross[e].std() + 0.012, f"{cross[e].mean():.3f}",
                ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [n for _, n in MODELS])
    style(ax, "sequence Macro-F1", (0.5, 1.02))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.set_title("Genre recognition: seen vs. unseen dancers (10 splits × 3 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_performance.png", dpi=180)
    plt.close(fig)

    dep = {e: seen[e].mean() - cross[e].mean() for e, _ in MODELS}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bar_ends(ax, x, [dep[e] for e, _ in MODELS], 0.6, BLUE)
    for i, (e, _) in enumerate(MODELS):
        ax.text(i, dep[e] + 0.006, f"{dep[e]:.3f}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(x, [n for _, n in MODELS])
    style(ax, "seen − unseen Macro-F1")
    ax.set_title("Dancer dependency (lower = less identity shortcut)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_dependency.png", dpi=180)
    plt.close(fig)
    notes += ["## 1–2. 성능 (cross / seen / 의존도)", "", "| 모델 | cross | seen | 의존도 |", "|---|---|---|---|"]
    notes += [f"| {n.replace(chr(10), ' ')} | {cross[e].mean():.3f} ± {cross[e].std():.3f} | "
              f"{seen[e].mean():.3f} | {dep[e]:.3f} |" for e, n in MODELS]
    a, b = cross["tf_aug_v2"], cross["tf_aug_adv_v2"]
    notes += ["", f"- +Aug+Adv 가 +Aug 보다 높은 반복: {int((b > a).sum())}/10, 평균 차이 {b.mean() - a.mean():+.3f}", ""]

    # 3: 세기 (시드 0 만, 같은 조건 비교)
    lam = [(0.0, "tf_aug_v2"), (0.25, "tf_aug_adv_lam0.25"), (0.5, "tf_aug_adv_v2"), (1.0, "tf_aug_adv_lam1.0")]
    lv = [(l, per_rep(e, "cross", seeds=[0]), per_rep(e, "seen", seeds=[0])) for l, e in lam]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs = [l for l, _, _ in lv]
    ys = [c.mean() for _, c, _ in lv]
    ax.plot(xs, ys, color=BLUE, lw=2, marker="o", ms=8, markeredgecolor="white", markeredgewidth=2, zorder=3)
    for xx, yy in zip(xs, ys):
        ax.text(xx, yy + 0.002, f"{yy:.3f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(xs, ["0\n(Aug only)", "0.25", "0.5\n(ours)", "1.0"])
    ax.set_xlabel("adversarial strength λ_max")
    style(ax, "unseen-dancer Macro-F1", (min(ys) - 0.015, max(ys) + 0.012))
    ax.set_title("Effect of adversarial strength (seed 0, 10 splits)")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_lambda.png", dpi=180)
    plt.close(fig)
    notes += ["## 3. 적대 학습 세기 (시드 0)", "", "| λ_max | cross | seen | 의존도 |", "|---|---|---|---|"]
    notes += [f"| {l} | {c.mean():.3f} | {s.mean():.3f} | {s.mean() - c.mean():.3f} |" for l, c, s in lv]
    notes.append("")

    # 4: probe
    nn_models = [m for m in MODELS if m[0] != "xgb"]
    pr = {e: per_rep(e, "cross", key=lambda r: r["probe"]["probe_acc"]) for e, _ in nn_models}
    chance = json.loads(next((R / "tf_plain_v2").glob("cross_*.json")).read_text(encoding="utf-8"))["probe"]["probe_chance"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xx = np.arange(len(nn_models))
    bar_ends(ax, xx, [pr[e].mean() for e, _ in nn_models], 0.6, BLUE, [pr[e].std() for e, _ in nn_models])
    for i, (e, _) in enumerate(nn_models):
        ax.text(i, pr[e].mean() + pr[e].std() + 0.01, f"{pr[e].mean():.3f}", ha="center", fontsize=10)
    ax.axhline(chance, color=GRAY, ls="--", lw=1.5, label=f"chance ({chance:.2f})")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xticks(xx, [n for _, n in nn_models])
    style(ax, "dancer-ID probe accuracy", (0.4, 1.0))
    ax.set_title("Dancer identity left in the representation (lower = better)")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_probe.png", dpi=180)
    plt.close(fig)
    notes += ["## 4. probe (표현에 남은 댄서 정보, 우연 수준 %.2f)" % chance, ""]
    notes += [f"- {n.replace(chr(10), ' ')}: {pr[e].mean():.3f}" for e, n in nn_models]
    notes.append("")

    # 5: 부위 제거
    parts = [("xgb", "all joints"), ("xgb_drop_head", "− head"), ("xgb_drop_arms", "− arms"), ("xgb_drop_legs", "− legs")]
    pv = {e: per_rep(e, "cross") for e, _ in parts}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xx = np.arange(len(parts))
    bar_ends(ax, xx, [pv[e].mean() for e, _ in parts], 0.6, [GRAY] + [BLUE] * 3, [pv[e].std() for e, _ in parts])
    for i, (e, _) in enumerate(parts):
        ax.text(i, pv[e].mean() + pv[e].std() + 0.01, f"{pv[e].mean():.3f}", ha="center", fontsize=10)
    ax.set_xticks(xx, [n for _, n in parts])
    style(ax, "unseen-dancer Macro-F1", (0.4, 0.8))
    ax.set_title("Body-part removal (XGBoost)")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_bodypart.png", dpi=180)
    plt.close(fig)
    notes += ["## 5. 부위 제거 (XGBoost, cross)", ""]
    notes += [f"- {n}: {pv[e].mean():.3f}" for e, n in parts]
    pg = {}
    for e, _ in parts:
        rows = [json.loads(f.read_text(encoding="utf-8"))["test"]["per_genre_f1"] for f in (R / e).glob("cross_rep*.json")]
        pg[e] = {g: np.mean([r[g] for r in rows]) for g in rows[0]}
    notes += ["", "| 장르 | 전체 | −머리 | −팔 | −다리 |", "|---|---|---|---|---|"]
    notes += [f"| {g} | " + " | ".join(f"{pg[e][g]:.2f}" for e, _ in parts) + " |" for g in pg["xgb"]]
    notes.append("")

    # 6: FineDance
    ext = defaultdict(list)
    for f in sorted((R / "external_finedance").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        ext[r["exp"]].append(r["external"])
    fd_models = [("xgb", "XGBoost"), ("tf_plain", "Transformer"), ("tf_adv", "+ Adv"),
                 ("tf_aug", "+ Aug"), ("tf_aug_adv", "+ Aug + Adv\n(ours)")]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(fd_models))
    for k, (mode, color, lab) in enumerate([("restricted", ORANGE, "choose among 5 shared genres"),
                                            ("open", BLUE, "choose among all 10 AIST++ genres")]):
        v = [[e[mode]["seq_macro_f1"] for e in ext[m]] for m, _ in fd_models]
        mean = [np.mean(a) for a in v]
        err = [np.std(a) if len(a) > 1 else 0 for a in v]
        bar_ends(ax, x + (k - 0.5) * w, mean, w, color, err, lab)
        for i in range(len(v)):
            ax.text(x[i] + (k - 0.5) * w, mean[i] + err[i] + 0.012, f"{mean[i]:.2f}", ha="center", fontsize=9)
    ax.axhline(0.2, color=GRAY, ls="--", lw=1.5, label="chance, 5 genres (0.2)")
    ax.set_xticks(x, [n for _, n in fd_models])
    style(ax, "FineDance Macro-F1", (0, 1.0))
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("External test: train on AIST++ → test on FineDance (3 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_finedance.png", dpi=180)
    plt.close(fig)
    notes += ["## 6. FineDance 외부 검증", "", "| 모델 | 10개 중 | 5개 중 | n |", "|---|---|---|---|"]
    for m, n in fd_models:
        o = [e["open"]["seq_macro_f1"] for e in ext[m]]
        r_ = [e["restricted"]["seq_macro_f1"] for e in ext[m]]
        notes.append(f"| {n.replace(chr(10), ' ')} | {np.mean(o):.3f} ± {np.std(o):.3f} | "
                     f"{np.mean(r_):.3f} ± {np.std(r_):.3f} | {len(o)} |")
    if "tf_aug_adv_raw" in ext:
        e = ext["tf_aug_adv_raw"][0]
        notes += ["", f"- 평활화 안 함 (tf_aug_adv, 시드 0): 10개 중 {e['open']['seq_macro_f1']:.3f}, "
                      f"5개 중 {e['restricted']['seq_macro_f1']:.3f}"]

    (OUT / "figures_numbers.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
