"""[4단계 · 팀원 작업] 혼동 행렬 분석과 장르 유사도 군집.

목표
1. results/<실험>/cross_rep*.json 의 test.confusion 을 모두 더해 행 정규화한 혼동 행렬 그림 저장
   (실험별 1장: xgb, tf_plain, tf_aug_adv) → results/analysis/confusion_<실험>.png
2. 가장 많이 헷갈리는 장르 쌍 상위 5개 표 (가설 H5: Middle Hip-hop–LA Hip-hop 이 1위인지)
3. 혼동 행렬을 대칭화해 거리로 바꾼 뒤 scipy 계층적 군집 덴드로그램 저장
   → Old School(BR, PO, LO, WA) / New School 이 데이터상에서도 나뉘는지 확인

json 구조: r["test"]["confusion"] 은 [정답][예측] 시퀀스 개수, 장르 순서는 r["test"]["per_genre_f1"] 키 순서.
"""
raise NotImplementedError("팀원 작업: 위 가이드대로 구현하세요.")
