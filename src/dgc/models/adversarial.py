"""장르 조건부 적대 학습.

왜 '장르 조건부'인가: AIST++에서는 댄서마다 장르가 하나뿐이라, 표현에서 '누가 췄는지'를 통째로 지우면
장르 정보도 함께 지워져 장르 분류와 정면으로 충돌한다. 그래서 판별기는 **같은 장르 안에서만** 댄서를
구분하게 한다 (장르마다 별도 출력 칸). 인코더는 gradient reversal을 통해 이 판별기를 속이도록 학습된다.
"""
import math

import torch
from torch import nn


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def dann_lambda(progress: float, gamma: float = 10.0) -> float:
    """Ganin & Lempitsky (2015)의 표준 스케줄: 0에서 시작해 1로 서서히 증가."""
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0


class GenreConditionalAdversary(nn.Module):
    def __init__(self, embed_dim, n_genres, k_max, valid_mask, hidden=128):
        """valid_mask: [G, K] bool. 장르 g의 학습 댄서 수가 K보다 적을 때 빈 칸을 막는다."""
        super().__init__()
        self.G, self.K = n_genres, k_max
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_genres * k_max))
        self.register_buffer("valid_mask", torch.as_tensor(valid_mask, dtype=torch.bool))

    def forward(self, z, genre, lam):
        r = GradReverse.apply(z, lam)
        logits = self.mlp(r).view(-1, self.G, self.K)
        sel = logits[torch.arange(z.shape[0], device=z.device), genre]       # [B, K]
        return sel.masked_fill(~self.valid_mask[genre], float("-inf"))
