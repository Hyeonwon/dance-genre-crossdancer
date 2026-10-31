"""표현 속 개인 정보량 측정 (linear probing).

학습된 인코더를 고정하고, 그 출력(임베딩)으로 '같은 장르 안에서 누가 췄는지'를 맞히는 선형 분류기를
학습 시퀀스로 학습 → 검증 시퀀스(같은 댄서들의 다른 시퀀스)로 평가한다.
정확도가 우연 수준(1/댄서 수)에 가까울수록 개인 정보가 지워졌다는 뜻.
"""
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def dancer_probe(Z_tr, g_tr, d_tr, Z_ev, g_ev, d_ev, max_per_genre=2000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    per_genre, chance = {}, {}
    for g in sorted(set(g_tr)):
        tr = np.where(g_tr == g)[0]
        ev = np.where(g_ev == g)[0]
        dancers = sorted(set(d_tr[tr]))
        if len(dancers) < 2 or len(ev) == 0:
            continue
        if len(tr) > max_per_genre:
            tr = rng.choice(tr, max_per_genre, replace=False)
        sc = StandardScaler().fit(Z_tr[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Z_tr[tr]), d_tr[tr])
        per_genre[str(g)] = float((clf.predict(sc.transform(Z_ev[ev])) == d_ev[ev]).mean())
        chance[str(g)] = 1.0 / len(dancers)
    if not per_genre:
        return {"probe_acc": None, "probe_chance": None, "per_genre": {}}
    return {"probe_acc": float(np.mean(list(per_genre.values()))),
            "probe_chance": float(np.mean(list(chance.values()))),
            "per_genre": per_genre}
