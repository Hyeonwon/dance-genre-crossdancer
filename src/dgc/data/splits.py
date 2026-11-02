"""데이터 분할.

cross (설정 A, 주 평가): 장르마다 댄서 1명을 무작위로 테스트로 빼고 나머지를 학습. n_repeats번 반복.
seen  (설정 B, 비교용):  시퀀스 단위 장르별 층화 무작위 분할. 테스트 비율은 cross와 같게 맞춘다.
두 설정 모두 학습 시퀀스의 val_frac을 장르별 층화로 떼어 검증셋으로 쓴다.
같은 시퀀스의 윈도우는 항상 같은 쪽에 있다 (분할 단위가 시퀀스).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def genre_dancer_table(seqs: dict) -> dict:
    table = defaultdict(lambda: defaultdict(int))
    for s in seqs.values():
        table[s["genre"]][s["dancer"]] += 1
    return {g: dict(sorted(d.items())) for g, d in sorted(table.items())}


def check_genre_dancer(seqs: dict):
    """장르별 댄서 수 확인 + 여러 장르를 춘 댄서가 있는지 경고."""
    table = genre_dancer_table(seqs)
    dancer_genres = defaultdict(set)
    for g, ds in table.items():
        for d in ds:
            dancer_genres[d].add(g)
    multi = {d: sorted(gs) for d, gs in dancer_genres.items() if len(gs) > 1}
    return table, multi


def _split_val(train: list, seqs: dict, val_frac: float, rng) -> tuple[list, list]:
    by_genre = defaultdict(list)
    for n in train:
        by_genre[seqs[n]["genre"]].append(n)
    tr, va = [], []
    for g in sorted(by_genre):
        names = sorted(by_genre[g])
        rng.shuffle(names)
        k = max(1, int(round(val_frac * len(names))))
        va += names[:k]
        tr += names[k:]
    return sorted(tr), sorted(va)


def make_cross_dancer(seqs: dict, n_repeats: int, val_frac: float, seed: int) -> list[dict]:
    table = genre_dancer_table(seqs)
    for g, ds in table.items():
        if len(ds) < 2:
            raise ValueError(f"{g} 장르의 댄서가 {len(ds)}명이라 댄서 기준 분할이 불가능합니다.")
    rng = np.random.default_rng(seed)
    splits = []
    for r in range(n_repeats):
        test_dancers = {g: str(rng.choice(sorted(ds))) for g, ds in table.items()}
        test = sorted(n for n, s in seqs.items() if test_dancers[s["genre"]] == s["dancer"])
        rest = [n for n in seqs if n not in set(test)]
        train, val = _split_val(rest, seqs, val_frac, rng)
        splits.append({"repeat": r, "train": train, "val": val, "test": test,
                       "test_dancers": test_dancers})
    return splits


def make_seen_dancer(seqs: dict, n_repeats: int, test_frac: float, val_frac: float,
                     seed: int) -> list[dict]:
    rng = np.random.default_rng(seed + 1000)
    by_genre = defaultdict(list)
    for n, s in seqs.items():
        by_genre[s["genre"]].append(n)
    splits = []
    for r in range(n_repeats):
        test = []
        for g in sorted(by_genre):
            names = sorted(by_genre[g])
            rng.shuffle(names)
            test += names[:int(round(test_frac * len(names)))]
        test = sorted(test)
        rest = [n for n in seqs if n not in set(test)]
        train, val = _split_val(rest, seqs, val_frac, rng)
        splits.append({"repeat": r, "train": train, "val": val, "test": test})
    return splits


def save_splits(splits: list, path: Path, split_cfg: dict, seq_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"split_cfg": split_cfg, "n_seqs": len(seq_names)}
    path.write_text(json.dumps({"meta": meta, "splits": splits}, indent=1), encoding="utf-8")


def load_splits(path: Path, cfg: dict, cache: dict) -> list:
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. 먼저 scripts/02_make_splits.py 를 실행하세요.")
    obj = json.loads(path.read_text(encoding="utf-8"))
    meta = obj["meta"]
    if meta["split_cfg"] != cfg["split"] or meta["n_seqs"] != len(cache["seqs"]):
        raise RuntimeError(f"{path} 가 현재 설정/캐시와 맞지 않습니다. python scripts/02_make_splits.py 를 다시 실행하세요.")
    return obj["splits"]
