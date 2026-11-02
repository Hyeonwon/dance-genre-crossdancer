"""데이터 증강: 뼈 길이 스케일링 (체형이 다른 '가상 댄서' 만들기)."""
import numpy as np

from dgc.data.preprocess import neck, pelvis
from dgc.data.skeleton import BONE_GROUPS, BONES


def bone_scale(W: np.ndarray, rng: np.random.Generator, max_scale: float) -> np.ndarray:
    """W: (L, 17, 3). 뼈 그룹마다 [1-s, 1+s] 배율을 뽑아 골반에서부터 다시 조립한다.
    좌우 뼈는 같은 그룹이라 같은 배율 → 몸이 비대칭으로 망가지지 않는다."""
    if max_scale <= 0:
        return W
    f = {g: rng.uniform(1 - max_scale, 1 + max_scale) for g in BONE_GROUPS}
    ext = np.concatenate([W, pelvis(W)[:, None], neck(W)[:, None]], axis=1)   # (L, 19, 3)
    new = ext.copy()
    for child, parent, group in BONES:
        new[:, child] = new[:, parent] + f[group] * (ext[:, child] - ext[:, parent])
    return new[:, :17]
