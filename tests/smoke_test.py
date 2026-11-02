"""가짜 데이터로 전체 파이프라인이 끝까지 도는지 확인 (Windows/Mac/Linux 공통).
    python tests/smoke_test.py
마지막에 "스모크 테스트 통과"가 나오면 OK. 중간 숫자는 가짜 데이터라 의미 없음.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
S = ["paths.anno_dir=data/fake_aist", "paths.cache=data/cache/fake.pkl", "paths.splits_dir=data/splits_fake",
     "paths.results_dir=results_fake", "split.n_repeats=2"]
FAST = ["nn.max_epochs=3", "nn.d_model=32", "nn.n_layers=1", "nn.ff_dim=64", "nn.stats_windows=300",
        "xgb.n_estimators=50", "xgb.max_depth_grid=[3]"]


def run(*args):
    cmd = [PY, *args]
    print("\n>>>", " ".join(str(a) for a in args), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"\n실패: {' '.join(str(a) for a in args)}  (위 에러 메시지 전체를 공유해 주세요)")


run("tests/make_fake_aist.py", "--out", "data/fake_aist")
run("scripts/00_inspect_data.py", "--limit", "30", "--set", *S)
run("scripts/01_build_cache.py", "--set", *S)
run("scripts/02_make_splits.py", "--set", *S)
for st in ("cross", "seen"):
    run("scripts/03_train_xgb.py", "--setting", st, "--repeats", "0", "--set", *S, *FAST)
    run("scripts/04_train_nn.py", "--exp", "tf_aug_adv", "--setting", st, "--repeats", "0", "--seeds", "0",
        "--set", *S, *FAST)
run("scripts/05_summarize.py", "--set", *S)
print("\n스모크 테스트 통과")
