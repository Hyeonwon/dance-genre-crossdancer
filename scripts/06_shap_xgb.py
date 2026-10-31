"""[4단계 · 팀원 작업] XGBoost 특징 해석 (SHAP).

목표: 장르별로 어떤 동작 특징이 판단에 중요한지 상위 10개를 뽑고 그림으로 저장한다.

구현 가이드
1. 03_train_xgb.py 와 같은 방식으로 cross 분할 rep0 에서 XGBoost를 학습한다
   (dgc.train.xgb.run_xgb 를 참고해 학습된 clf 를 반환하도록 함수를 하나 더 만들어도 된다).
2. `pip install shap` 후 shap.TreeExplainer(clf) 로 테스트 윈도우의 SHAP 값을 구한다.
   멀티클래스라 결과는 (윈도우, 특징, 장르) 모양이다.
3. 장르마다 |SHAP| 평균 상위 10개 특징을 표로 저장 → results/analysis/shap_top10.md
4. 장르별 막대그래프를 results/analysis/shap_<장르>.png 로 저장.
5. 가설 H3 확인용: 다리 관련 특징이 BR·HO 에서, 팔 관련 특징이 PO·LO·WA 에서 상위에 오는지 메모.

특징 이름은 dgc.features.handcrafted.feature_names(keep) 에 있다.
"""
raise NotImplementedError("팀원 작업: 위 가이드대로 구현하세요.")
