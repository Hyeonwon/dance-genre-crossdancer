"""scripts/ 에서 src/dgc 를 설치 없이 import 할 수 있게 경로 추가."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
