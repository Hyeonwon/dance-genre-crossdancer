"""백본 등록. 새 백본(예: MotionBERT)을 추가하려면 같은 인터페이스로 만들고 여기에 등록한다.

인터페이스: model(x) -> (genre_logits [B, G], embedding [B, D]),  model.embed_dim = D
x: [B, L, F] (윈도우 프레임별 특징, 표준화 전 값. 표준화는 모델 안에서 한다)
"""
from dgc.models.transformer import DanceTransformer


def build_backbone(cfg, in_dim: int, n_genres: int, max_len: int):
    name = cfg["nn"]["backbone"]
    if name == "transformer":
        n = cfg["nn"]
        return DanceTransformer(in_dim, n_genres, n["d_model"], n["n_layers"], n["n_heads"],
                                n["ff_dim"], n["dropout"], max_len)
    raise ValueError(f"등록되지 않은 백본: {name}")
