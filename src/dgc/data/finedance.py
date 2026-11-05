"""FineDance 외부 검증용 로더.

FineDance(ICCV 2023)는 AIST++와 다른 댄서·다른 스튜디오·다른 모션캡처 장비로 만든 데이터다.
AIST++와 겹치는 장르만 골라 AIST++ 캐시와 같은 형식으로 만든다. **학습에는 절대 섞지 않는다.**

변환 과정
1. motion/*.npy (T, 315) = 골반 위치 3 + SMPL-H 52관절 6D 회전 (30fps, y축 위)
2. 몸통 22관절만 FK → COCO 17관절 근사 (smpl_fk.py)
3. AIST++에서 추정한 관절별 보정값 적용 (configs/smpl_calib.json, scripts/08_calibrate_smpl.py)
4. 댄서 정보가 없으므로 시퀀스마다 자기 몸통 길이로 정규화 (dancer = 시퀀스 이름)
"""
import json
import pickle
from pathlib import Path

import numpy as np

from dgc.config import resolve
from dgc.data.preprocess import floor_height, torso_length
from dgc.data.smpl_fk import smplh_rot6d_fk, to_coco17

# FineDance 세부 장르(style2, 소문자) → AIST++ 장르 코드. 정답으로 인정하는 코드 목록.
# Hiphop은 AIST++에서 Middle/LA 두 장르로 나뉘어 있어 둘 다 정답으로 인정한다.
# 대응이 애매한 장르(Urban, Korean, 중국 고전·민속 등)는 넣지 않는다.
GENRE_MAP: dict[str, list[str]] = {
    "breaking": ["gBR"],
    "popping": ["gPO"],
    "locking": ["gLO"],
    "hiphop": ["gMH", "gLH"],
    "jazz": ["gJS"],          # FineDance의 Street 계열 Jazz → AIST++ Street Jazz
}
FD_LABELS = list(GENRE_MAP)


def find_root(path) -> Path:
    """압축을 푼 위치에 따라 한 단계 더 안쪽에 폴더가 생기는 경우를 처리."""
    p = resolve(path)
    for cand in [p, *p.glob("*")]:
        if (cand / "motion").is_dir() and (cand / "label_json").is_dir():
            return cand
    raise FileNotFoundError(f"{p} 아래에서 motion/ 과 label_json/ 폴더를 찾지 못했습니다.")


def build_finedance_cache(cfg) -> dict:
    fd = cfg["finedance"]
    root = find_root(fd["root"])
    calib = np.array(json.loads(resolve(fd["calib"]).read_text(encoding="utf-8"))["calib_m"])
    up = cfg["data"]["up_axis"]
    seqs, torso, skipped = {}, {}, {}
    for jf in sorted((root / "label_json").glob("*.json")):
        lab = json.loads(jf.read_text(encoding="utf-8"))
        style = str(lab.get("style2", "")).strip().lower()
        mf = root / "motion" / f"{jf.stem}.npy"
        if style not in GENRE_MAP or not mf.exists():
            skipped[style] = skipped.get(style, 0) + 1
            continue
        P, G = smplh_rot6d_fk(np.load(mf))
        J = to_coco17(P, G, calib)
        if fd.get("src_fps", 30) != cfg["data"]["fps"]:
            J = J[:: fd["src_fps"] // cfg["data"]["fps"]]
        name = f"fd_{jf.stem}"
        seqs[name] = {"name": name, "genre": GENRE_MAP[style][0], "fd_label": style,
                      "accept": GENRE_MAP[style], "situation": "fd", "dancer": name,
                      "song": lab.get("name"), "joints": J, "floor": floor_height(J, up)}
        torso[name] = torso_length(J)
    cache = {"seqs": seqs, "torso": torso, "fps": cfg["data"]["fps"], "up_axis": up,
             "skipped_styles": skipped}
    out = resolve(fd["cache"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(cache, f)
    counts = {}
    for s in seqs.values():
        counts[s["fd_label"]] = counts.get(s["fd_label"], 0) + 1
    print(f"FineDance 캐시 저장: {out} ({len(seqs)}개 시퀀스) 장르별 {counts}")
    print(f"  제외한 장르: {skipped}")
    return cache


def load_finedance_cache(cfg, rebuild=False) -> dict:
    p = resolve(cfg["finedance"]["cache"])
    if rebuild or not p.exists():
        return build_finedance_cache(cfg)
    with open(p, "rb") as f:
        return pickle.load(f)
