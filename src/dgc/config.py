"""설정 파일 로드: configs/default.yaml + configs/exp/<exp>.yaml + 명령줄 --set 덮어쓰기."""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _set_path(cfg: dict, dotted: str, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(exp: str | None = None, overrides=()) -> dict:
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml", encoding="utf-8"))
    if exp:
        exp_path = ROOT / "configs" / "exp" / f"{exp}.yaml"
        cfg = _deep_update(cfg, yaml.safe_load(open(exp_path, encoding="utf-8")))
        cfg["exp_name"] = exp
    for o in overrides or ():
        key, val = o.split("=", 1)
        _set_path(cfg, key, yaml.safe_load(val))
    return cfg


def resolve(path: str) -> Path:
    """설정 속 상대경로는 레포 루트 기준."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def window_len(cfg) -> int:
    return int(round(cfg["data"]["window_sec"] * cfg["data"]["fps"]))


def window_stride(cfg) -> int:
    return int(round(cfg["data"]["stride_sec"] * cfg["data"]["fps"]))
