"""[4단계] XGBoost 특징 해석 (SHAP): 장르마다 어떤 동작 특징·신체 부위로 판단하는가.

- SHAP 값은 XGBoost 내장 TreeSHAP(pred_contribs)으로 정확히 계산한다 (shap 패키지 불필요).
- cross 분할의 반복마다 XGBoost를 다시 학습하고(03과 같은 설정, 그 반복에서 고른 max_depth 사용),
  처음 보는 댄서(테스트) 윈도우에서 '정답 장르 점수'에 대한 SHAP 값을 모은다.
- 특징을 신체 부위(머리·팔·다리·전신)로 묶어 부위별 기여 비율도 낸다 → 가설 H3 확인, 부위 제거 실험과 비교.

실행:
  pip install matplotlib        # 그림용 (한 번만)
  python scripts/06_shap_xgb.py                 # 반복 10개 전부 (CPU 10~20분)
  python scripts/06_shap_xgb.py --repeats 0,1   # 빠르게 확인
결과: results/analysis/shap_top10.md, shap_bodypart.md, shap_top10.png, shap_bodypart.png
"""
import argparse
import json
import time
from collections import defaultdict

import _bootstrap  # noqa: F401
import numpy as np
import xgboost as xgb

from dgc.config import load_config, resolve
from dgc.data.naming import GENRE_NAMES
from dgc.data.preprocess import load_cache
from dgc.data.skeleton import JOINT_GROUPS, keep_joints
from dgc.data.splits import load_splits
from dgc.data.windows import genre_index, handcrafted_for_all
from dgc.features.handcrafted import JOINT_NAMES, feature_names
from dgc.train.common import parse_repeats, result_path
from dgc.train.xgb import _gather

PARTS = ["head", "arms", "legs", "global"]


def body_part(feat: str) -> str:
    for g, joints in JOINT_GROUPS.items():
        if any(feat.startswith(JOINT_NAMES[j] + "_") for j in joints):
            return g
    return "global"


def fit_xgb(cfg, Xtr, ytr, Xva, yva, depth, seed):
    x = cfg["xgb"]
    clf = xgb.XGBClassifier(n_estimators=x["n_estimators"], max_depth=depth,
                            learning_rate=x["learning_rate"], tree_method="hist",
                            early_stopping_rounds=x["early_stopping_rounds"],
                            eval_metric="mlogloss", n_jobs=x["n_jobs"], random_state=seed)
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return clf


def true_class_shap(clf, X, y):
    """윈도우마다 '정답 장르' 점수에 대한 특징별 SHAP 값 (n, D)."""
    booster = clf.get_booster()
    it = (0, clf.best_iteration + 1) if getattr(clf, "best_iteration", None) is not None else (0, 0)
    contrib = booster.predict(xgb.DMatrix(X), pred_contribs=True, iteration_range=it)  # (n, C, D+1)
    return contrib[np.arange(len(y)), y, :-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.set)
    cache = load_cache(cfg)
    splits = load_splits(resolve(cfg["paths"]["splits_dir"]) / "cross.json", cfg, cache)
    g2i = genre_index(cache)
    i2g = {i: g for g, i in g2i.items()}
    names = feature_names(keep_joints(cfg["data"]["drop_joint_groups"]))
    parts = np.array([body_part(f) for f in names])

    t0 = time.time()
    feats = handcrafted_for_all(cache, cfg)
    print(f"수작업 특징 계산 완료 ({time.time() - t0:.0f}s), 특징 {len(names)}개")

    per_genre = defaultdict(list)       # genre -> [(n, D) |SHAP|]
    for r in parse_repeats(args.repeats, len(splits)):
        sp = splits[r]
        prev = result_path(cfg, "xgb", "cross", r)
        depth = json.loads(prev.read_text(encoding="utf-8"))["best_max_depth"] if prev.exists() \
            else cfg["xgb"]["max_depth_grid"][1]
        Xtr, ytr, _ = _gather(feats, cache, sp["train"], g2i)
        Xva, yva, _ = _gather(feats, cache, sp["val"], g2i)
        Xte, yte, _ = _gather(feats, cache, sp["test"], g2i)
        clf = fit_xgb(cfg, Xtr, ytr, Xva, yva, depth, seed=r)
        S = np.abs(true_class_shap(clf, Xte, yte))
        for c in np.unique(yte):
            per_genre[i2g[int(c)]].append(S[yte == c])
        print(f"  rep{r}: depth {depth}, 테스트 윈도우 {len(yte)}개 ({time.time() - t0:.0f}s)")

    genres = [g for g in g2i if g in per_genre]
    imp = {g: np.concatenate(per_genre[g]).mean(0) for g in genres}        # (D,) 평균 |SHAP|
    share = {g: np.array([imp[g][parts == p].sum() for p in PARTS]) / imp[g].sum() for g in genres}

    out = resolve(cfg["paths"]["results_dir"]) / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    # 1) 장르별 상위 10개 특징
    lines = ["# 장르별 중요 특징 상위 10개 (XGBoost, 처음 보는 댄서 테스트 윈도우의 평균 |SHAP|)", "",
             "정답 장르 점수에 대한 기여 크기. 괄호 안은 신체 부위.", ""]
    for g in genres:
        top = np.argsort(-imp[g])[:10]
        lines += [f"## {GENRE_NAMES[g]} ({g})", "", "| 순위 | 특징 | 부위 | 평균 \\|SHAP\\| |", "|---|---|---|---|"]
        lines += [f"| {k + 1} | {names[j]} | {parts[j]} | {imp[g][j]:.3f} |" for k, j in enumerate(top)]
        lines.append("")
    (out / "shap_top10.md").write_text("\n".join(lines), encoding="utf-8")

    # 2) 신체 부위별 기여 비율 (가설 H3)
    lines = ["# 신체 부위별 기여 비율 (장르별 |SHAP| 합의 비율, %)", "",
             "| 장르 | " + " | ".join(PARTS) + " |", "|---|" + "---|" * len(PARTS)]
    lines += [f"| {GENRE_NAMES[g]} | " + " | ".join(f"{100 * v:.0f}" for v in share[g]) + " |" for g in genres]
    leg = {g: share[g][PARTS.index("legs")] for g in genres}
    arm = {g: share[g][PARTS.index("arms")] for g in genres}
    lines += ["", "## 가설 H3 확인",
              f"- 다리 비중 상위 3개 장르: {', '.join(GENRE_NAMES[g] for g in sorted(leg, key=lambda g: -leg[g])[:3])}",
              f"- 팔 비중 상위 3개 장르: {', '.join(GENRE_NAMES[g] for g in sorted(arm, key=lambda g: -arm[g])[:3])}",
              "- H3 예상: 다리 → Break·House, 팔 → Pop·Lock·Waack"]
    (out / "shap_bodypart.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # 3) 그림
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[알림] matplotlib 이 없어 그림은 건너뜀 → pip install matplotlib 후 다시 실행")
        return
    colors = {"head": "#8c6bb1", "arms": "#e6550d", "legs": "#3182bd", "global": "#969696"}
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for ax, g in zip(axes.ravel(), genres):
        top = np.argsort(-imp[g])[:10][::-1]
        ax.barh([names[j] for j in top], imp[g][top], color=[colors[parts[j]] for j in top])
        ax.set_title(GENRE_NAMES[g], fontsize=13)
        ax.tick_params(axis="y", labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[p]) for p in PARTS]
    fig.legend(handles, PARTS, loc="lower center", ncol=4, fontsize=12)
    fig.suptitle("Top-10 features per genre (mean |SHAP|, unseen-dancer test windows)", fontsize=15)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out / "shap_top10.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 6))
    M = np.stack([share[g] for g in genres]) * 100
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max())
    ax.set_xticks(range(len(PARTS)), PARTS)
    ax.set_yticks(range(len(genres)), [GENRE_NAMES[g] for g in genres])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                    color="white" if M[i, j] > M.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, label="share of |SHAP| (%)")
    ax.set_title("Body-part contribution per genre (XGBoost SHAP)")
    fig.tight_layout()
    fig.savefig(out / "shap_bodypart.png", dpi=150)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
