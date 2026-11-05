"""[4단계] 혼동 행렬 분석과 장르 유사도 군집.

1. results/<실험>/cross_rep*.json 의 test.confusion 을 모두 더해 행 정규화 → confusion_<실험>.png
2. 가장 많이 헷갈리는 장르 쌍 상위 5개 → confused_pairs_<실험>.md (가설 H5: Middle–LA Hip-hop 이 1위인지)
3. 혼동을 대칭화해 거리로 바꾼 뒤 계층적 군집 덴드로그램 → genre_dendrogram_<실험>.png
   (Old School: Break·Pop·Lock·Waack 이 한쪽으로 모이는지)

사용법:
  python scripts/07_confusion_analysis.py                       # xgb, tf_plain_v2, tf_aug_adv_v2
  python scripts/07_confusion_analysis.py --exp tf_aug_v2
"""
import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np

from dgc.config import resolve
from dgc.data.naming import GENRE_NAMES

SHORT = {"gBR": "Break", "gPO": "Pop", "gLO": "Lock", "gWA": "Waack", "gMH": "M-HipHop",
         "gLH": "LA-HipHop", "gHO": "House", "gKR": "Krump", "gJS": "S-Jazz", "gJB": "B-Jazz"}


def load_confusion(exp):
    files = sorted(resolve(f"results/{exp}").glob("cross_rep*.json"))
    if not files:
        raise FileNotFoundError(f"results/{exp}/cross_rep*.json 이 없습니다.")
    mats, genres = [], None
    for f in files:
        t = json.loads(f.read_text(encoding="utf-8"))["test"]
        mats.append(np.array(t["confusion"], dtype=float))
        genres = list(t["per_genre_f1"])
    return np.sum(mats, 0), genres, len(files)


def plot(exp, M, genres, n_files, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    P = M / M.sum(1, keepdims=True)
    lab = [SHORT[g] for g in genres]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(genres)), lab, rotation=45, ha="right")
    ax.set_yticks(range(len(genres)), lab)
    for i in range(len(genres)):
        for j in range(len(genres)):
            if P[i, j] >= 0.005:
                ax.text(j, i, f"{100 * P[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if P[i, j] > 0.55 else "#0b0b0b")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{exp}: unseen-dancer confusion (%, {n_files} runs pooled)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / f"confusion_{exp}.png", dpi=150)
    plt.close(fig)

    S = (P + P.T) / 2
    np.fill_diagonal(S, 0)
    D = 1 - S / S.max()          # 가장 많이 헷갈린 쌍 = 거리 0
    np.fill_diagonal(D, 0)
    fig, ax = plt.subplots(figsize=(7, 4))
    dendrogram(linkage(squareform(D, checks=False), "average"), labels=lab, ax=ax,
               color_threshold=0, above_threshold_color="#2a78d6")
    ax.set_ylabel("distance (1 − normalized confusion)")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"{exp}: genre similarity from confusions")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / f"genre_dendrogram_{exp}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=["xgb", "tf_plain_v2", "tf_aug_adv_v2"])
    args = ap.parse_args()
    out = resolve("results/analysis")
    out.mkdir(parents=True, exist_ok=True)
    for exp in args.exp:
        M, genres, n = load_confusion(exp)
        P = M / M.sum(1, keepdims=True)
        pairs = sorted(((P[i, j] + P[j, i], i, j) for i in range(len(genres))
                        for j in range(i + 1, len(genres))), reverse=True)[:5]
        lines = [f"# 가장 많이 헷갈린 장르 쌍 ({exp}, 처음 보는 댄서, {n}회 합산)", "",
                 "| 순위 | 장르 쌍 | A→B | B→A | 합 |", "|---|---|---|---|---|"]
        for k, (s, i, j) in enumerate(pairs):
            lines.append(f"| {k + 1} | {GENRE_NAMES[genres[i]]} ↔ {GENRE_NAMES[genres[j]]} | "
                         f"{100 * P[i, j]:.0f}% | {100 * P[j, i]:.0f}% | {100 * s:.0f}% |")
        (out / f"confused_pairs_{exp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines) + "\n")
        try:
            plot(exp, M, genres, n, out)
        except ImportError as e:
            print(f"[알림] 그림 건너뜀 ({e}) → pip install matplotlib scipy")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
