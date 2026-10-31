"""Transformer 인코더 분류기: 프레임 특징 → 선형 투영 → [CLS] + 학습형 위치 임베딩 → 인코더 → 장르."""
import torch
from torch import nn


class DanceTransformer(nn.Module):
    def __init__(self, in_dim, n_genres, d_model=128, n_layers=4, n_heads=4, ff_dim=256,
                 dropout=0.1, max_len=120):
        super().__init__()
        self.embed_dim = d_model
        # 입력 표준화 통계 (학습 데이터로 set_input_stats에서 채움)
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.inp = nn.Linear(in_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_genres)

    def set_input_stats(self, mean, std):
        self.x_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.x_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6))

    def embed(self, x):
        x = (x - self.x_mean) / self.x_std
        h = self.inp(x)
        h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], dim=1)
        h = h + self.pos[:, :h.shape[1]]
        h = self.encoder(h)
        return self.norm(h[:, 0])

    def forward(self, x):
        z = self.embed(x)
        return self.head(z), z
