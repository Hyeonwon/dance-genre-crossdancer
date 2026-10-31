#!/usr/bin/env bash
# 전체 실험 한 번에 돌리기 (GPU 권장). 먼저 빠른 확인용으로 REPEATS=0,1,2 SEEDS=0 정도로 돌려 보세요.
#   REPEATS=0,1,2 SEEDS=0 bash scripts/run_all.sh
set -e
REPEATS=${REPEATS:-}
SEEDS=${SEEDS:-}
R=${REPEATS:+--repeats $REPEATS}
S=${SEEDS:+--seeds $SEEDS}

python scripts/01_build_cache.py
python scripts/02_make_splits.py

for setting in cross seen; do
  python scripts/03_train_xgb.py --setting $setting $R
  for exp in tf_plain tf_aug tf_adv tf_aug_adv; do
    python scripts/04_train_nn.py --exp $exp --setting $setting $R $S
  done
done

# 신체 부위 제거 실험 (cross만)
for g in head arms legs; do
  python scripts/03_train_xgb.py --setting cross $R --tag drop_$g --set "data.drop_joint_groups=[$g]"
done

python scripts/05_summarize.py
