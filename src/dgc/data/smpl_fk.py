"""SMPL 계열(SMPL / SMPL-H) 회전값 → 관절 좌표 (순운동학, FK) → COCO 17관절 근사.

AIST++(SMPL 72차원 축-각)와 FineDance(SMPL-H 52관절 6D 회전)를 **같은 뼈대**로 관절 좌표로 바꾼다.
두 데이터셋이 같은 뼈대·같은 관절 정의를 쓰게 되므로, 데이터셋 차이가 '관절 정의 차이'가 아니라
'춤 자체의 차이'만 남는다. SMPL 모델 파일(라이선스 필요) 없이 평균 체형의 고정 뼈 길이를 쓴다.
(체형 차이는 어차피 전처리에서 몸통 길이로 나누고, 뼈 길이 증강으로 다룬다.)

좌표계: SMPL 기본 자세 기준 +x = 몸의 왼쪽, +y = 위, +z = 앞.
"""
import numpy as np

# SMPL 몸통 22관절의 부모 (SMPL-H도 앞 22개가 같다. 22~51은 손가락이라 쓰지 않음)
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# 평균 체형 SMPL의 부모→자식 뼈 벡터 (m). EDGE(Tseng et al., 2023) 시각화 코드의 값과 같다.
OFFSETS = np.array([
    [0.0, 0.0, 0.0],
    [0.05858135, -0.08228004, -0.01766408],
    [-0.06030973, -0.09051332, -0.01354254],
    [0.00443945, 0.12440352, -0.03838522],
    [0.04345142, -0.38646945, 0.008037],
    [-0.04325663, -0.38368791, -0.00484304],
    [0.00448844, 0.1379564, 0.02682033],
    [-0.01479032, -0.42687458, -0.037428],
    [0.01905555, -0.4200455, -0.03456167],
    [-0.00226458, 0.05603239, 0.00285505],
    [0.04105436, -0.06028581, 0.12204243],
    [-0.03483987, -0.06210566, 0.13032329],
    [-0.0133902, 0.21163553, -0.03346758],
    [0.07170245, 0.11399969, -0.01889817],
    [-0.08295366, 0.11247234, -0.02370739],
    [0.01011321, 0.08893734, 0.05040987],
    [0.12292141, 0.04520509, -0.019046],
    [-0.11322832, 0.04685326, -0.00847207],
    [0.2553319, -0.01564902, -0.02294649],
    [-0.26012748, -0.01436928, -0.03126873],
    [0.26570925, 0.01269811, -0.00737473],
    [-0.26910836, 0.00679372, -0.00602676],
], dtype=np.float64)

HEAD = 15
# 얼굴 점(코·눈·귀)은 SMPL 관절에 없으므로 머리 관절 기준 고정 위치로 둔다 (머리와 함께 회전)
FACE_OFFSETS = np.array([
    [0.0, 0.0, 0.11],      # nose
    [0.03, 0.035, 0.09],   # l_eye
    [-0.03, 0.035, 0.09],  # r_eye
    [0.075, 0.0, 0.0],     # l_ear
    [-0.075, 0.0, 0.0],    # r_ear
])
# COCO 5~16 ← SMPL 관절 번호
COCO_FROM_SMPL = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
# COCO 관절마다 '붙어 움직이는' SMPL 관절 (보정값을 이 관절의 좌표계로 표현). 엉덩이 점은 골반에 붙인다.
ATTACH = [HEAD] * 5 + [16, 17, 18, 19, 20, 21, 0, 0, 4, 5, 7, 8]


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """(..., 3) 축-각 → (..., 3, 3) 회전 행렬 (로드리게스 공식)."""
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(theta, 1e-8)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    K = np.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], -1).reshape(aa.shape[:-1] + (3, 3))
    th = theta[..., None]
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """(..., 6) 6D 회전 → (..., 3, 3). pytorch3d 규칙(앞 두 '행')을 따른다 (FineDance/EDGE와 동일)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def forward_kinematics(R: np.ndarray, trans: np.ndarray):
    """R: (T, 22, 3, 3) 관절별 지역 회전, trans: (T, 3) 골반 위치
    → 관절 좌표 (T, 22, 3), 관절별 전역 회전 (T, 22, 3, 3)."""
    T = R.shape[0]
    G = np.zeros((T, 22, 3, 3))
    P = np.zeros((T, 22, 3))
    for j, p in enumerate(PARENTS):
        if p < 0:
            G[:, j], P[:, j] = R[:, j], trans
        else:
            G[:, j] = G[:, p] @ R[:, j]
            P[:, j] = P[:, p] + (G[:, p] @ OFFSETS[j])
    return P, G


def to_coco17(P: np.ndarray, G: np.ndarray, calib=None) -> np.ndarray:
    """SMPL 22관절 좌표 → COCO 17관절 (T, 17, 3).
    calib: (17, 3) 관절별 보정 벡터 (붙은 관절 좌표계, m). AIST++의 실제 COCO 좌표와 짝지어 추정한 값
    (scripts/08_calibrate_smpl.py). SMPL 관절(뼈 회전 중심)과 COCO 점(몸 표면)의 정의 차이를 메운다."""
    face = P[:, HEAD, None, :] + np.einsum("tij,kj->tki", G[:, HEAD], FACE_OFFSETS)
    J = np.concatenate([face, P[:, COCO_FROM_SMPL]], axis=1)
    if calib is not None:
        J = J + np.einsum("tkij,kj->tki", G[:, ATTACH], np.asarray(calib))
    return J.astype(np.float32)


def smpl_axis_angle_fk(poses: np.ndarray, trans: np.ndarray):
    """AIST++ motions: poses (T, 72) 축-각, trans (T, 3) m 단위 → FK 결과 (P, G)."""
    R = axis_angle_to_matrix(poses.reshape(-1, 24, 3)[:, :22].astype(np.float64))
    return forward_kinematics(R, trans.astype(np.float64))


def smplh_rot6d_fk(motion: np.ndarray):
    """FineDance motion: (T, 315) = 골반 위치 3 + 52관절 × 6D → FK 결과 (P, G)."""
    trans = motion[:, :3].astype(np.float64)
    R = rot6d_to_matrix(motion[:, 3:].reshape(-1, 52, 6)[:, :22].astype(np.float64))
    return forward_kinematics(R, trans)
