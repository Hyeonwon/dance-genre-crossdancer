"""[확장 · 형원 작업] FineDance 외부 검증용 로더.

목표: FineDance 중 AIST++와 겹치는 장르만 골라, AIST++ 캐시와 같은 형식으로 만든다.
그러면 학습된 모델을 그대로 불러 테스트만 하면 된다 (학습에는 절대 섞지 않는다).

해야 할 일
1. FineDance 장르 목록을 확인하고 AIST++ 10개 장르와의 대응표를 만든다 (GENRE_MAP).
   애매한 장르는 넣지 않는다. 겹치는 장르 수가 적으면 팀원과 상의해서 범위를 정한다.
2. FineDance 모션(SMPL-H)을 smplx 로 관절 좌표로 바꾼 뒤 COCO 17관절에 대응시킨다.
   (SMPL 관절에는 눈·귀가 없으므로 머리 관절은 head 관절 위치로 근사하거나, 머리 그룹을 빼고
    data.drop_joint_groups=[head] 로 학습한 모델과 비교한다.)
3. fps 를 30으로 맞추고, 수직축 방향을 AIST++ 와 같게 맞춘다.
4. build_cache 와 같은 구조의 dict 를 반환한다:
   {"seqs": {name: {"genre", "dancer", "situation", "joints", "floor"}}, "torso": {dancer: float},
    "fps": 30, "up_axis": ...}
"""

GENRE_MAP: dict[str, str] = {
    # "FineDance 장르 이름": "gXX",   ← 확인 후 채우기
}


def build_finedance_cache(cfg):
    raise NotImplementedError("확장 작업: 위 설명대로 구현하세요.")
