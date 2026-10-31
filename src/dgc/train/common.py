import hashlib
import json
from pathlib import Path

import numpy as np

from dgc.config import resolve


def result_path(cfg, exp: str, setting: str, rep: int, seed: int | None = None) -> Path:
    """파일 이름에 전처리 버전을 넣어, 설정을 바꿔 다시 돌려도 예전 결과를 덮어쓰지 않게 한다."""
    name = f"{setting}_rep{rep:02d}" + (f"_seed{seed}" if seed is not None else "") + f"_v{prep_version(cfg)}.json"
    return resolve(cfg["paths"]["results_dir"]) / exp / name


def save_result(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=_default))


def parse_repeats(arg: str | None, n: int) -> list[int]:
    if not arg:
        return list(range(n))
    return [int(x) for x in arg.split(",")]


# 결과 비교에 영향을 주는 설정. 이 값들이 같아야 결과끼리 비교할 수 있다.
# (drop_joint_groups는 실험 조건이라 제외: 부위 제거 실험도 같은 버전으로 비교되어야 함)
def prep_settings(cfg) -> dict:
    d = {k: v for k, v in cfg["data"].items() if k != "drop_joint_groups"}
    sp = {k: cfg["split"][k] for k in ("n_repeats", "val_frac", "seed")}
    return {"data": d, "split": sp}


def prep_version(cfg) -> str:
    blob = json.dumps(prep_settings(cfg), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]
