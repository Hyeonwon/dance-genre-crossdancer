# 팀원 작업 가이드

> 위에서부터 순서대로 따라 하면 됩니다. 각 단계 끝에 **"완료 기준"** 이 있어요. 거기까지 되면 다음 단계로 넘어가세요.
> 막히면 **30분 넘게 붙잡지 말고** 바로 형원에게 물어보세요 (보낼 내용은 맨 아래 참고).

## 프로젝트 한 줄 요약

춤 동작(3D 관절 좌표)만 보고 장르를 맞히는 모델을 만드는데, 모델이 장르 대신 **"누가 췄는지"를 외워서 맞히는 꼼수**를 쓰는지 확인하고 막는 프로젝트예요.

- **팀원 담당:** 데이터 확인 → XGBoost 기준선 실험(꼼수가 있는지 확인) → 혼동 행렬 그림 → 발표용 표·그래프
- **형원 담당:** Transformer와 적대 학습(꼼수를 막는 방법), SHAP, 발표

## 일정

| 기한 | 할 일 |
|---|---|
| **11/8 (일)** | 0~4단계: 환경 준비, 데이터 확인, XGBoost 실험 |
| **11/15 (일)** | 5단계: 혼동 행렬 분석 |
| **11/22 (일)** | 6단계: 발표용 그래프 7장 (지금 시작 가능) |

**11/8 결과가 제일 중요해요.** 이 숫자를 보고 프로젝트 방향을 확정하거든요.

---

## 0단계. 환경 준비 (30분)

필요한 것: Python 3.10 이상, git. **GPU는 필요 없어요** (XGBoost는 CPU로 충분).

```bash
git clone https://github.com/Hyeonwon/dance-genre-crossdancer.git
cd dance-genre-crossdancer

# 가상환경 (권장)
python -m venv .venv
source .venv/bin/activate          # Mac/Linux

pip install -r requirements.txt
pip install matplotlib scipy        # 5~6단계 그림용

# 가짜 데이터로 전체가 도는지 확인 (몇 분)
python tests/smoke_test.py
```

**Windows (PowerShell)** 는 가상환경 활성화가 달라요:
```powershell
python -m venv .venv                 # 'python'이 없다고 나오면 py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1          # 줄 앞에 (.venv) 가 붙으면 성공
```
데이터 압축 해제는 파일 탐색기로 해도 돼요. 최종 폴더 구조만 1단계 그림과 같으면 됩니다.

**완료 기준:** 마지막 줄에 `스모크 테스트 통과`가 나오면 OK. 중간 숫자는 가짜 데이터라 의미 없어요.

---

## 1단계. 데이터 다운로드 (30분)

AIST++ 사이트의 예전 다운로드 링크는 만료돼서, 파일이 **GitHub 릴리스 페이지**로 옮겨졌어요.
(AIST++ 사이트 https://google.github.io/aistplusplus_dataset 의 다운로드 안내를 따라가면 나와요. 주소는 형원에게 받아도 됨)

릴리스 페이지의 Assets 에서 **딱 두 개만** 받으세요.

| 파일 | 크기 | 받기 |
|---|---|---|
| `keypoints3d.zip` | 836MB | ✅ |
| `ignore_list.txt` | 1KB | ✅ |
| 나머지 (keypoints2d, motions, cameras, splits) | | ❌ 안 받아도 됨 |

레포 안에 넣기:
```bash
mkdir -p data/aist_plusplus_final
unzip ~/Downloads/keypoints3d.zip -d data/aist_plusplus_final/
cp ~/Downloads/ignore_list.txt data/aist_plusplus_final/
```

최종 구조는 이렇게 되어야 해요. 압축을 풀었을 때 폴더가 한 단계 더 안쪽에 생기면 옮겨서 맞춰 주세요.
```
dance-genre-crossdancer/
└── data/
    └── aist_plusplus_final/
        ├── keypoints3d/        ← .pkl 파일 1,400개 정도
        └── ignore_list.txt
```

확인:
```bash
ls data/aist_plusplus_final/keypoints3d | head -3
ls data/aist_plusplus_final/keypoints3d | wc -l
```

**완료 기준:** `gBR_sBM_cAll_d04_mBR0_ch01.pkl` 같은 이름의 파일이 보이고, 개수가 1,400개 안팎이면 OK.

---

## 2단계. 데이터 확인 → 건너뛰어도 됨

형원이 이미 확인을 끝냈어요 (`results/data_report.md`). 장르당 댄서 3명, 수직축 설정 일치 → **설정 그대로 사용**.
궁금하면 `python scripts/00_inspect_data.py` 를 돌려서 같은 결과가 나오는지 봐도 되지만 필수는 아니에요.

---

## 3단계. 캐시와 분할 만들기 (10~20분)

```bash
python scripts/01_build_cache.py
python scripts/02_make_splits.py
```

**완료 기준:** `02_make_splits.py` 출력에 아래 두 가지가 보이면 OK.
- 장르 10개, 각 장르에 댄서 3명씩
- `cross: 10회`, `seen: 10회`

---

## 4단계. XGBoost 실험 (실행만 하면 됨, 1~2시간)

아래 명령을 **순서대로** 실행하세요. 노트북을 켜 두고 다른 일 해도 돼요.

```bash
# (1) 처음 보는 댄서로 테스트
python scripts/03_train_xgb.py --setting cross

# (2) 아는 댄서로 테스트 (비교용)
python scripts/03_train_xgb.py --setting seen

# (3) 신체 부위를 하나씩 빼고 테스트
python scripts/03_train_xgb.py --setting cross --tag drop_head --set "data.drop_joint_groups=[head]"
python scripts/03_train_xgb.py --setting cross --tag drop_arms --set "data.drop_joint_groups=[arms]"
python scripts/03_train_xgb.py --setting cross --tag drop_legs --set "data.drop_joint_groups=[legs]"

# (4) 요약
python scripts/05_summarize.py
```

`05_summarize.py` 가 표 두 개를 출력해요. 이 중 **"댄서 의존도"** 표가 핵심이에요.

```
| 실험 | seen | cross | 의존도 |
| xgb  | 0.xxx | 0.xxx | +0.xxx |
```
- `seen` 이 `cross` 보다 높을수록 (의존도가 클수록) 모델이 댄서를 외우는 꼼수를 많이 쓴다는 뜻이에요.

### 결과 올리기 (git)

```bash
git checkout -b exp/xgb-baseline
git add results/
git commit -m "XGBoost 기준선 결과 (cross/seen + 부위 제거) + 데이터 보고서"
git push -u origin exp/xgb-baseline
```
그다음 GitHub 레포 페이지에서 **Compare & pull request** 버튼을 눌러 PR을 만들어 주세요. `data/` 폴더는 자동으로 빠지니 신경 안 써도 돼요.

**완료 기준:** PR 생성 + `05_summarize.py` 출력의 두 표를 카톡으로 형원에게 공유. **(11/8까지)**

---

## 5단계. 혼동 행렬 분석 → ✅ 완료 (형원, `scripts/07_confusion_analysis.py`)

`scripts/07_confusion_analysis.py` 를 구현해요. 파일 안에도 설명이 있어요. **"어떤 장르끼리 헷갈리는지"** 를 그림과 표로 보여주는 작업이에요.

### 만들 것
| 파일 | 내용 |
|---|---|
| `results/analysis/confusion_xgb.png` | 혼동 행렬 히트맵 (행: 정답, 열: 예측, 각 행 합이 1이 되게 정규화) |
| `results/analysis/confused_pairs.md` | 가장 많이 헷갈린 장르 쌍 상위 5개 표 |
| `results/analysis/genre_dendrogram.png` | 장르 유사도 덴드로그램 |

### 구현 순서 (힌트)
```python
import json, glob
import numpy as np

# 1) cross 결과 10개를 모두 읽어서 혼동 행렬을 더한다
files = glob.glob("results/xgb/cross_rep*.json")
mats, genres = [], None
for f in files:
    r = json.load(open(f))
    mats.append(np.array(r["test"]["confusion"]))
    genres = list(r["test"]["per_genre_f1"].keys())   # 장르 순서 (gBR, gPO, ...)
M = np.sum(mats, axis=0)

# 2) 행 정규화
P = M / M.sum(axis=1, keepdims=True)

# 3) matplotlib imshow로 히트맵, 칸마다 숫자 표시, 축 라벨은 genres
# 4) 헷갈린 쌍: i != j 인 칸에서 P[i,j] + P[j,i] 가 큰 순서로 5개
# 5) 덴드로그램: D = 1 - (P + P.T)/2 (대각선 0) → scipy.cluster.hierarchy.linkage(squareform(D), "average")
```
장르 코드를 이름으로 바꿀 때는 `src/dgc/data/naming.py` 의 `GENRE_NAMES` 를 쓰면 돼요.

### 보고 메모할 것
- 가장 많이 헷갈린 쌍이 **Middle Hip-hop ↔ LA Hip-hop** 인가?
- 덴드로그램에서 Old School(Break, Pop, Lock, Waack)이 한쪽으로 모이는가?

**완료 기준:** 위 그림·표 3개 + 메모 두 줄을 브랜치 `feat/confusion-analysis` 로 PR.

나중에 형원 결과가 올라오면 `--exp tf_aug_adv` 처럼 실험 이름만 바꿔서 같은 그림을 한 장 더 만들 수 있게, 실험 이름을 인자로 받도록 만들면 좋아요 (`results/xgb/` 대신 `results/<실험이름>/`).

---

## 6단계. 발표용 그래프 → ✅ 완료 (형원, `scripts/10_figures.py`, 결과 `results/analysis/`)

형원 실험이 **전부 끝나서 지금 바로 시작해도 돼요.**

```bash
git checkout main
git pull
```

`scripts/10_figures.py` 를 새로 만들어서, 아래 그림을 `results/analysis/` 에 png로 저장해요.
숫자는 전부 `results/` 안의 json 에서 읽어요.

### 읽을 결과 폴더 (중요)
| 폴더 | 내용 | 파일 |
|---|---|---|
| `xgb` | XGBoost | `cross_rep00 ~ 09`, `seen_rep00 ~ 09` |
| `tf_plain_v2`, `tf_aug_v2`, `tf_adv_v2`, `tf_aug_adv_v2` | Transformer 4종 | 반복 10개 × 시드 3개 (`_seed0`, `_seed1`, `_seed2`) |
| `tf_aug_adv_lam0.25`, `tf_aug_adv_lam1.0` | 적대 학습 세기 실험 | 시드 0만 |
| `xgb_drop_head/arms/legs` | 부위 제거 | cross 10개 |
| `external_finedance` | 다른 데이터셋(FineDance) 시험 | 실험별 시드 0~2 |

- `tf_plain`, `tf_aug_adv` (뒤에 `_v2` 없는 폴더)는 **예전 버전이라 쓰지 마세요.**
- 시드가 여러 개인 경우: **같은 반복(rep)끼리 시드 평균** → 그다음 반복 10개의 평균·표준편차.
- 점수: `test.seq_macro_f1`. FineDance만 `external.open.seq_macro_f1` (10개 장르 중 고르기) 와
  `external.restricted.seq_macro_f1` (겹치는 5개 중 고르기).

### 만들 그림
1. **실험별 성능 막대그래프:** x축 = xgb, tf_plain, tf_adv, tf_aug, tf_aug_adv (`_v2`), 막대 2개(cross, seen), 오차 막대 = 표준편차
2. **댄서 의존도 막대그래프:** 실험별 (seen − cross)
3. **적대 학습 세기 그래프 (꺾은선):** x축 = 세기 0, 0.25, 0.5, 1.0 / y축 = cross Macro-F1. 모두 **시드 0** 파일만 사용
   - 0 → `tf_aug_v2`, 0.25 → `tf_aug_adv_lam0.25`, 0.5 → `tf_aug_adv_v2`, 1.0 → `tf_aug_adv_lam1.0`
4. **개인 정보 그래프:** 실험별 `probe.probe_acc` 평균 (cross 결과에서). 가로선으로 `probe.probe_chance`(0.5) 표시
5. **부위 제거 그래프:** xgb vs drop_head / drop_arms / drop_legs 의 cross Macro-F1
6. **FineDance 그래프:** x축 = 실험 5개, 막대 2개(open, restricted), 오차 막대 = 시드 표준편차 (xgb는 1개라 막대 없음). `_raw` 결과는 빼기
7. 5단계 혼동 행렬을 `tf_aug_adv_v2` 로도 한 장

참고 숫자 (그래프가 맞게 나왔는지 확인용, cross Macro-F1):
xgb 0.667 / tf_plain 0.788 / tf_adv 0.795 / tf_aug 0.847 / tf_aug_adv 0.857

**완료 기준:** 그림 7장 + `10_figures.py` 를 브랜치 `feat/presentation-figures` 로 PR.

---

## 하지 말 것

- `configs/default.yaml` 수정 (설정이 달라지면 서로 결과를 비교할 수 없게 됨. 바꿔야 할 것 같으면 형원에게 먼저 말하기)
- `src/` 아래 코드 수정 (07 스크립트만 새로 작성)
- `main` 브랜치에 바로 push (항상 브랜치 → PR)
- `results/` 안의 다른 사람 결과 삭제

## 막혔을 때 보낼 것

1. 실행한 명령 그대로
2. 에러 메시지 **전체** (마지막 줄만 말고 `Traceback` 부터 끝까지)
3. 몇 단계에서 막혔는지

## 용어 정리

| 용어 | 뜻 |
|---|---|
| cross | 처음 보는 댄서로 테스트하는 설정 (주 평가) |
| seen | 학습에서 본 댄서로 테스트하는 설정 (비교용) |
| 댄서 의존도 | seen 성능 − cross 성능. 클수록 "누가 췄는지"를 외우는 꼼수가 크다 |
| Macro-F1 | 장르별 F1의 평균. 장르마다 고르게 잘 맞히는지 보는 지표 |
| 혼동 행렬 | 정답 장르별로 모델이 어떤 장르로 예측했는지 센 표 |
| rep | 반복 번호. 테스트 댄서를 바꿔 가며 10번 실험 |
