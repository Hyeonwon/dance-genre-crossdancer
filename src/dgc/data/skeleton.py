"""COCO 17관절 정의.

0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear,
5 l_shoulder, 6 r_shoulder, 7 l_elbow, 8 r_elbow, 9 l_wrist, 10 r_wrist,
11 l_hip, 12 r_hip, 13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle
"""

N_JOINTS = 17
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE = 5, 6, 11, 12, 15, 16

JOINT_GROUPS = {
    "head": [0, 1, 2, 3, 4],
    "arms": [5, 6, 7, 8, 9, 10],
    "legs": [11, 12, 13, 14, 15, 16],
}
UPPER = list(range(0, 11))
LOWER = list(range(11, 17))
MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

# 뼈 길이 증강용 트리. 17 = 골반(엉덩이 중점), 18 = 목(어깨 중점)인 가상 관절.
# (child, parent, 뼈 그룹) — 부모가 항상 먼저 나오도록 정렬. 좌우는 같은 그룹 = 같은 배율.
BONES = [
    (11, 17, "hip"), (12, 17, "hip"), (18, 17, "spine"),
    (5, 18, "shoulder"), (6, 18, "shoulder"), (0, 18, "neck"),
    (1, 0, "head"), (2, 0, "head"), (3, 1, "head"), (4, 2, "head"),
    (7, 5, "upper_arm"), (8, 6, "upper_arm"), (9, 7, "forearm"), (10, 8, "forearm"),
    (13, 11, "thigh"), (14, 12, "thigh"), (15, 13, "shin"), (16, 14, "shin"),
]
BONE_GROUPS = sorted({g for _, _, g in BONES})


def keep_joints(drop_groups) -> list[int]:
    drop = set()
    for g in drop_groups or []:
        drop |= set(JOINT_GROUPS[g])
    return [j for j in range(N_JOINTS) if j not in drop]
