"""AIST++ 시퀀스 이름 파싱.

예: gBR_sBM_cAll_d04_mBR0_ch01
  g = 장르, s = 상황(sBM=Basic, sFM=Advanced), c = 카메라, d = 댄서, m = 음악, ch = 안무
"""

GENRES = ["gBR", "gPO", "gLO", "gWA", "gMH", "gLH", "gHO", "gKR", "gJS", "gJB"]
GENRE_NAMES = {
    "gBR": "Break", "gPO": "Pop", "gLO": "Lock", "gWA": "Waack",
    "gMH": "Middle Hip-hop", "gLH": "LA Hip-hop", "gHO": "House",
    "gKR": "Krump", "gJS": "Street Jazz", "gJB": "Ballet Jazz",
}


def parse_seq_name(name: str) -> dict:
    parts = name.split("_")
    if len(parts) != 6:
        raise ValueError(f"예상과 다른 시퀀스 이름 형식: {name}")
    genre, situation, camera, dancer, music, choreo = parts
    if not (genre.startswith("g") and dancer.startswith("d") and situation.startswith("s")):
        raise ValueError(f"예상과 다른 시퀀스 이름 형식: {name}")
    return {
        "name": name, "genre": genre, "situation": situation, "camera": camera,
        "dancer": dancer, "music": music, "choreo": choreo,
    }
