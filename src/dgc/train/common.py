import json
from pathlib import Path

import numpy as np

from dgc.config import resolve


def result_path(cfg, exp: str, setting: str, rep: int, seed: int | None = None) -> Path:
    name = f"{setting}_rep{rep:02d}" + (f"_seed{seed}" if seed is not None else "") + ".json"
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
