"""[1단계] 전체 시퀀스를 읽어 30fps 캐시 생성.  실행: python scripts/01_build_cache.py"""
import argparse

import _bootstrap  # noqa: F401

from dgc.config import load_config
from dgc.data.preprocess import build_cache

ap = argparse.ArgumentParser()
ap.add_argument("--set", nargs="*", default=[], help="설정 덮어쓰기 예: data.up_axis=2")
args = ap.parse_args()
build_cache(load_config(overrides=args.set))
