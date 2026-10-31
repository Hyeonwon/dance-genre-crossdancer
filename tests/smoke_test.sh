#!/usr/bin/env bash
# 가짜 데이터로 전체 파이프라인이 끝까지 도는지 확인 (CPU로 몇 분). 코드 수정 후 PR 전에 돌려 보세요.
#   bash tests/smoke_test.sh
set -e
cd "$(dirname "$0")/.."
python tests/make_fake_aist.py --out data/fake_aist
S="paths.anno_dir=data/fake_aist paths.cache=data/cache/fake.pkl paths.splits_dir=data/splits_fake paths.results_dir=results_fake split.n_repeats=2"
FAST="nn.max_epochs=3 nn.d_model=32 nn.n_layers=1 nn.ff_dim=64 nn.stats_windows=300 xgb.n_estimators=50 xgb.max_depth_grid=[3]"
python scripts/00_inspect_data.py --set $S --limit 30
python scripts/01_build_cache.py --set $S
python scripts/02_make_splits.py --set $S
for st in cross seen; do
  python scripts/03_train_xgb.py --setting $st --repeats 0 --set $S $FAST
  python scripts/04_train_nn.py --exp tf_aug_adv --setting $st --repeats 0 --seeds 0 --set $S $FAST
done
python scripts/05_summarize.py --set $S
echo "스모크 테스트 통과"
