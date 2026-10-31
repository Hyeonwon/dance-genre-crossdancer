# 처음 보는 댄서의 춤도 장르를 맞힐 수 있는가

빅데이터 입문 기말 프로젝트 (2026 Fall)

AIST++ 3D 관절 좌표로 10개 춤 장르를 분류한다. AIST++는 장르마다 댄서가 3명이고 한 댄서가 한 장르만 춘다. 그래서 모델이 장르 대신 **"누가 췄는지"를 외우는 꼼수(shortcut learning)** 를 써도 정답을 맞힐 수 있다. 이 프로젝트에서 하는 일은 세 가지다.

1. **확인:** 처음 보는 댄서로 테스트(cross-subject)해서 꼼수가 얼마나 큰지 잰다.
2. **해결:** Transformer에 **장르 조건부 적대 학습**과 **뼈 길이 증강**을 붙여 꼼수를 막는다.
3. **해석:** 꼼수를 걷어낸 뒤에도 남는 장르별 핵심 동작을 찾는다.

---

## 빠른 시작

```bash
git clone https://github.com/Hyeonwon/dance-genre-crossdancer.git
cd dance-genre-crossdancer
pip install -r requirements.txt

# 1) 진짜 데이터 없이 전체가 도는지 먼저 확인 (CPU 몇 분)
bash tests/smoke_test.sh
```

### 데이터 준비
1. AIST++ 공식 사이트(google.github.io/aistplusplus_dataset)에서 **annotations**를 받는다. 영상은 필요 없다.
2. 압축을 풀어서 `keypoints3d/` 폴더와 `ignore_list.txt`가 들어 있는 폴더를 `data/aist_plusplus_final/` 에 둔다.
   - 다른 위치에 두려면 `configs/default.yaml`의 `paths.anno_dir` 를 수정하면 된다.
3. `data/`와 `results/`는 git에 올리지 않는다(.gitignore). 각자 로컬에서 만든다.

### 실행 순서

| 단계 | 명령 | 결과 |
|---|---|---|
| 0 | `python scripts/00_inspect_data.py` | `results/data_report.md` (장르별 댄서 구성, 수직축 확인) |
| 1 | `python scripts/01_build_cache.py` | `data/cache/aist_30fps.pkl` |
| 1 | `python scripts/02_make_splits.py` | `data/splits/cross.json`, `seen.json` |
| 2 | `python scripts/03_train_xgb.py --setting cross` (그리고 `seen`) | `results/xgb/*.json` |
| 3 | `python scripts/04_train_nn.py --exp tf_plain --setting cross` | `results/tf_plain/*.json` |
| 4 | `python scripts/05_summarize.py` | `results/summary.md` |

전체를 한 번에 돌리려면 `bash scripts/run_all.sh`. 처음에는 `REPEATS=0,1,2 SEEDS=0 bash scripts/run_all.sh` 처럼 줄여서 돌리는 걸 추천한다.

설정은 `--set` 으로 덮어쓸 수 있다. 예: `--set nn.lr=5e-4 adv.weight=0.5`

---

## 실험 구성

| 이름 | 내용 | 역할 |
|---|---|---|
| `xgb` | XGBoost + 수작업 특징 약 93개 | 꼼수가 있는지 빠르게 **확인**, 특징 해석 |
| `tf_plain` | Transformer 인코더 | 강한 모델도 꼼수를 부린다는 걸 **보여줌** |
| `tf_aug` | + 증강 (뼈 길이 ±15%, 좌우 반전, 속도 ±10%) | 증강만의 효과 |
| `tf_adv` | + 장르 조건부 적대 학습 | 적대 학습만의 효과 |
| `tf_aug_adv` | + 증강 + 적대 학습 | **제안 방법** |
| `xgb_drop_{head,arms,legs}` | 신체 부위 하나를 빼고 학습 | 장르별 핵심 신체 부위 |

### 분할 (`src/dgc/data/splits.py`)
- **cross (주 평가):** 장르마다 댄서 1명을 무작위로 테스트로 뺀다(학습 20명 / 테스트 10명). 이걸 10번 반복한다.
- **seen (비교):** 시퀀스 단위 무작위 분할. 테스트 비율은 cross와 같게 맞춘다.
- **댄서 의존도 = seen 성능 − cross 성능.** 이 값이 꼼수의 크기다.

### 평가 지표 (`src/dgc/eval/`)
- **주 지표:** 시퀀스 단위 Macro-F1. 시퀀스 예측은 윈도우 확률의 평균으로 정한다.
- **Basic(sBM) / Advanced(sFM)** 별 F1도 따로 낸다.
- **개인 정보량(probe):** 학습된 표현으로 "같은 장르 안에서 누가 췄는지"를 맞히는 선형 분류기의 정확도. 우연 수준에 가까울수록 개인 정보가 지워졌다는 뜻이다.

### 왜 "장르 조건부" 적대 학습인가 (`src/dgc/models/adversarial.py`)
댄서마다 장르가 하나뿐이라, 표현에서 "누가 췄는지"를 통째로 지우면 장르 정보까지 같이 지워진다. 그래서 판별기는 **같은 장르 안에서만** 댄서를 구분하도록 만든다. 인코더는 gradient reversal을 통해 이 판별기를 속이도록 학습된다.

---

## 폴더 구조

```
configs/
  default.yaml          모든 설정 (경로, 전처리, 모델, 증강, 적대 학습)
  exp/*.yaml            실험별 차이만
src/dgc/
  data/                 이름 파싱, 로딩, 전처리·정규화, 증강, 분할, 윈도우
  features/             XGBoost용 수작업 특징
  models/               Transformer 인코더, 적대 학습(GRL + 장르별 판별기)
  train/                XGBoost / 신경망 학습 루프
  eval/                 지표, probe
scripts/                00~07 실행 스크립트
tests/                  가짜 데이터 생성, 스모크 테스트
```

---

## 역할 분담 (제안)

두 사람의 기여도가 같아야 하니까, 각자 "자기 결과"가 남는 단위로 나눈다.

형원이 방법 설계·구현·튜닝과 발표를 맡고, 팀원은 **이미 만들어진 스크립트를 실행하고 결과를 정리하는 일** 위주로 맡는다. 팀원 작업도 각자 결과물(보고서, 표, 그림)이 남도록 나눴다.

| 단계 | 형원 | 팀원 |
|---|---|---|
| 1. 데이터 | 레포·파이프라인 세팅 (완료) | 데이터 다운로드, `00`~`02` 실행, 데이터 보고서 공유 |
| 2. 기준선 | 결과 검토 | XGBoost 실험 실행 (명령만 실행하면 됨, 아래 체크리스트) |
| 3. 제안 방법 | Transformer 4종 실험, 적대 학습 튜닝 | — |
| 4. 해석 | probe 분석, SHAP(`06`) | 혼동 행렬 그림·헷갈리는 장르 표(`07`), 결과 표·그래프 정리 |
| 확장 | FineDance 외부 검증 | — |
| 발표 | 발표 (전체) | 발표 자료용 그림·표 준비 |

> **팀원은 [docs/TEAMMATE_GUIDE.md](docs/TEAMMATE_GUIDE.md) 를 따라 하면 됩니다** (단계별 명령, 완료 기준, 기한).

### 팀원 체크리스트 (요약)
**1주차 (~11/8): 데이터 확인 + 기준선**
- [ ] 레포 클론 → `pip install -r requirements.txt` → `bash tests/smoke_test.sh` 가 "스모크 테스트 통과"로 끝나는지 확인
- [ ] AIST++ annotations 다운로드 → 압축 풀어서 `data/aist_plusplus_final/` 에 두기
- [ ] `python scripts/00_inspect_data.py` → 만들어진 `results/data_report.md` 를 형원에게 공유
- [ ] `python scripts/01_build_cache.py` → `python scripts/02_make_splits.py`
- [ ] 아래 명령을 차례로 실행하고, 마지막에 나오는 표를 형원에게 공유
  ```bash
  python scripts/03_train_xgb.py --setting cross
  python scripts/03_train_xgb.py --setting seen
  python scripts/03_train_xgb.py --setting cross --tag drop_head --set "data.drop_joint_groups=[head]"
  python scripts/03_train_xgb.py --setting cross --tag drop_arms --set "data.drop_joint_groups=[arms]"
  python scripts/03_train_xgb.py --setting cross --tag drop_legs --set "data.drop_joint_groups=[legs]"
  python scripts/05_summarize.py
  ```

**2~3주차: 그림과 표**
- [ ] `scripts/07_confusion_analysis.py` 구현 (파일 안에 단계별 가이드 있음. matplotlib로 그림 그리는 수준)
- [ ] 형원이 Transformer 결과를 올리면 `05_summarize.py` 결과를 발표용 표·막대그래프로 정리

막히면 바로 물어보기. 혼자 30분 넘게 붙잡고 있지 않기.

### 작업 규칙
- 작업마다 브랜치를 만들고 PR로 합친다. 예: `feat/confusion-analysis`
- PR 전에 `bash tests/smoke_test.sh` 가 통과하는지 확인한다.
- **결과 공유:** 실험이 끝나면 `results/` 안의 json(과 그림)을 커밋해서 올린다. 각자 다른 실험 폴더(`results/xgb*/`, `results/tf_*/`)에 쓰니까 충돌하지 않는다. `results/summary.md` 는 올리지 않고 각자 `05_summarize.py` 로 만든다.
- 분할 파일(`data/splits/`)은 같은 데이터·같은 seed면 누가 만들어도 같다.

### 전처리 설정을 바꿔도 결과가 섞이지 않게 해 뒀다
- 결과 파일마다 **전처리 버전**(`_v1a2b3c4d.json`)이 붙는다. `data:` 와 `split:` 설정으로 계산한 값이라, 설정이 같으면 버전도 같다.
- `05_summarize.py` 는 버전별로 따로 묶고, 기본으로 **현재 설정 버전**만 보여준다(`--all` 로 전부). 예전 결과를 지울 필요 없다.
- 캐시를 만들 때 쓰는 설정(`up_axis`, `up_sign`, `fps`, `src_fps`, `keypoints_key`)을 바꾸고 캐시를 다시 안 만들면, 실행이 멈추고 다시 만들라는 메시지가 나온다.
- 전처리를 바꿨다면 **상대방에게 알리기**. 비교하려면 XGBoost와 Transformer 둘 다 같은 버전으로 다시 돌려야 한다.

---

## 알아 둘 점

- **가짜 데이터의 숫자는 의미 없다.** 스모크 테스트는 코드가 끝까지 도는지만 확인한다.
- **증강을 켜면 수렴이 느려진다.** 가짜 데이터에서도 epoch가 2~3배 더 필요했다. `nn.max_epochs` 와 `patience` 를 줄이지 말 것.
- **적대 학습은 판별기 세기가 중요하다.** 판별기가 약하면 인코더가 쉽게 속이고 끝나서 개인 정보가 실제로는 남는다(probe로 확인). 그래서 판별기 학습률 배수 `adv.lr_mult`(기본 10)를 따로 뒀다. 실제 데이터에서 `adv.weight` 와 함께 튜닝해야 한다.
- 실행 시간: 데이터가 작아서 GPU 한 장(Colab 무료 포함)이면 충분하다. 전체 조합(10회 × 시드 3 × 실험 4 × 설정 2)은 오래 걸리니, 먼저 반복 3회 × 시드 1로 경향을 보고 늘린다.
- 다른 백본(예: MotionBERT)을 비교하려면 `src/dgc/models/__init__.py` 의 인터페이스대로 추가한다.

## 참고
- Li et al. (2021) *AI Choreographer: Music Conditioned 3D Dance Generation with AIST++.* ICCV.
- Ganin & Lempitsky (2015) *Unsupervised Domain Adaptation by Backpropagation.* ICML.
- Gulrajani & Lopez-Paz (2021) *In Search of Lost Domain Generalization.* ICLR.
- Geirhos et al. (2020) *Shortcut Learning in Deep Neural Networks.* Nature Machine Intelligence.
