"""scripts/ 에서 src/dgc 를 설치 없이 import 할 수 있게 경로 추가."""
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):   # Windows 터미널(cp949)에서 한글·기호 출력 오류 방지
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
