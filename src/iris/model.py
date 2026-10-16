import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from iris.config import ModelConfig


class IRIS(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        hidden_dim: int,
        dropout: float,
        output_dim: int,
        ini_queries: torch.Tensor | np.ndarray,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.tensor(ini_queries, dtype=torch.float32))
        self.head_dim = d_model // nhead
        self.num_heads = nhead
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.p = dropout
        self.nhead = nhead

    def query_cross_attention(self, x: torch.Tensor, attn_mask) -> torch.Tensor:
        N, D = self.queries.size()  # (num_queries, d_model)
        B, L, D = x.size()  # (batch_size, seq_len, d_model)
        E = D // self.nhead
        H = self.nhead
        # (batch_size, num_queries, d_model)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        # (batch_size, nhead, num_queries, head_dim)
        queries = queries.view(B, N, H, E).transpose(1, 2)
        # (batch_size, nhead, seq_len, head_dim)
        x = x.view(B, L, H, E).transpose(1, 2)
        # (batch_size, 1, 1, seq_len)
        attn_mask = attn_mask.view(B, 1, 1, L)
        # (batch_size, nhead, num_queries, head_dim)
        out = F.scaled_dot_product_attention(
            query=queries,
            key=x,
            value=x,
            attn_mask=attn_mask,
            dropout_p=(self.p if self.training else 0.0),
        )
        # Attention returns (batch_size, nhead, num_queries, head_dim).
        # Move the query dimension back before joining the attention heads.
        out = out.transpose(1, 2).reshape(B, N, D)
        return out

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        out = self.query_cross_attention(x, attn_mask=padding_mask)
        out = self.mlp(out)  # (batch_size, num_queries, dim_feedforward)
        out = out.mean(dim=1)  # (batch_size, d_model)
        out = self.fc(out)
        return out


def make_model(ini_queries: np.ndarray, cfg: ModelConfig):
    exclude = {"num_queries", "query_ini_random", "auto_device_map", "device"}
    model_config = cfg.model_dump(exclude=exclude)
    return IRIS(ini_queries=ini_queries, **model_config)
