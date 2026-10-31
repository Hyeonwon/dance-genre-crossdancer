"""XGBoost 기준선: 꼼수(댄서 의존)가 있는지, 얼마나 큰지 빠르게 확인하는 진단 도구."""
import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from dgc.data.skeleton import keep_joints
from dgc.data.windows import genre_index
from dgc.eval.metrics import compute_metrics, seq_aggregate
from dgc.features.handcrafted import feature_names


def _gather(feats, cache, names, g2i):
    X, y, seq = [], [], []
    for n in names:
        if n not in feats:
            continue
        X.append(feats[n])
        y += [g2i[cache["seqs"][n]["genre"]]] * len(feats[n])
        seq += [n] * len(feats[n])
    return np.concatenate(X), np.array(y), seq


def run_xgb(cfg, cache, feats, split, seed=0) -> dict:
    g2i = genre_index(cache)
    Xtr, ytr, _ = _gather(feats, cache, split["train"], g2i)
    Xva, yva, sva = _gather(feats, cache, split["val"], g2i)
    Xte, yte, ste = _gather(feats, cache, split["test"], g2i)
    x = cfg["xgb"]
    best = None
    for depth in x["max_depth_grid"]:
        clf = XGBClassifier(n_estimators=x["n_estimators"], max_depth=depth,
                            learning_rate=x["learning_rate"], tree_method="hist",
                            early_stopping_rounds=x["early_stopping_rounds"],
                            eval_metric="mlogloss", n_jobs=x["n_jobs"], random_state=seed)
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        names, p = seq_aggregate(clf.predict_proba(Xva), sva)
        yv = np.array([g2i[cache["seqs"][n]["genre"]] for n in names])
        f1 = f1_score(yv, p.argmax(1), labels=list(range(len(g2i))), average="macro", zero_division=0)
        if best is None or f1 > best[0]:
            best = (f1, depth, clf)
    val_f1, depth, clf = best
    test = compute_metrics(clf.predict_proba(Xte), yte, ste, cache, g2i)
    fn = feature_names(keep_joints(cfg["data"]["drop_joint_groups"]))
    imp = clf.feature_importances_
    top = sorted(zip(fn, imp.tolist()), key=lambda t: -t[1])[:20]
    return {"model": "xgb", "best_max_depth": depth, "val_seq_macro_f1": float(val_f1),
            "best_iteration": int(clf.best_iteration), "test": test, "top_features_gain": top,
            "n_train_windows": len(ytr), "n_test_windows": len(yte)}
