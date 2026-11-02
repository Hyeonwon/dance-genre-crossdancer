"""AIST++ annotations 읽기 (keypoints3d/*.pkl, ignore_list.txt)."""
import pickle
from pathlib import Path

import numpy as np


def load_ignore_list(anno_dir: Path) -> set:
    p = Path(anno_dir) / "ignore_list.txt"
    if not p.exists():
        print(f"[경고] {p} 가 없습니다. 제외 목록 없이 진행합니다.")
        return set()
    return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


def list_sequences(anno_dir: Path) -> list[str]:
    kp_dir = Path(anno_dir) / "keypoints3d"
    if not kp_dir.exists():
        raise FileNotFoundError(f"{kp_dir} 가 없습니다. configs/default.yaml 의 paths.anno_dir 를 확인하세요.")
    return sorted(p.stem for p in kp_dir.glob("*.pkl"))


def fill_nans(arr: np.ndarray) -> np.ndarray:
    """(T, J, 3)에서 NaN을 시간축 선형 보간으로 채운다. 전부 NaN이면 0."""
    arr = arr.copy()
    T = arr.shape[0]
    t = np.arange(T)
    flat = arr.reshape(T, -1)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        bad = ~np.isfinite(col)
        if bad.all():
            col[:] = 0.0
        elif bad.any():
            col[bad] = np.interp(t[bad], t[~bad], col[~bad])
    return flat.reshape(arr.shape)


def load_keypoints3d(anno_dir: Path, seq_name: str, key: str = "keypoints3d") -> np.ndarray:
    with open(Path(anno_dir) / "keypoints3d" / f"{seq_name}.pkl", "rb") as f:
        d = pickle.load(f)
    arr = np.asarray(d[key], dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (17, 3):
        raise ValueError(f"{seq_name}: 예상 모양 (N,17,3), 실제 {arr.shape}")
    return fill_nans(arr)
