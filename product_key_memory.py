from typing import Dict, Tuple, List, Any
import torch
import torch.nn as nn
import math


def fetch_pkm_value_parameters(
    module: nn.Module,
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    params: List[nn.Parameter] = []
    for m in module.modules():
        if isinstance(m, PKM):
            params.append(m.values.weight)  # type: ignore
    paramset = set(params)
    rest = [p for p in module.parameters() if p not in paramset]
    return params, rest


def fetch_optimizer_parameters(
    module: nn.Module, pkm_learning_rate: float = 1e-2
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pkm_params, rest = fetch_pkm_value_parameters(module)
    return {"params": rest}, {"params": pkm_params, "lr": pkm_learning_rate}


class PKM(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        num_keys: int = 128,
        topk: int = 32,
        dim_head: int = 256,
        input_dropout: float = 0.0,
        query_dropout: float = 0.0,
        value_dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % heads == 0, "dimension must be divisible by number of heads"
        self.topk = topk
        self.heads = heads
        self.num_keys = num_keys

        dim_query = dim_head * heads
        self.to_queries = nn.Linear(dim, dim_query, bias=False)
        self.norm = nn.LayerNorm(dim_query)

        self.keys = nn.Parameter(torch.zeros(heads, num_keys, 2, dim_head // 2))
        self.values = nn.EmbeddingBag(num_keys ** 2, dim, mode="sum")

        self.input_dropout = nn.Dropout(input_dropout)
        self.query_dropout = nn.Dropout(query_dropout)
        self.value_dropout = nn.Dropout(value_dropout)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.keys, std=1 / self.keys.size(-1))
        nn.init.normal_(
            self.values.weight, std=1 / math.sqrt(self.values.weight.size(-1))
        )

    def forward(self, x, input_mask=None, **kwargs):
        t, b, e = x.shape
        h = self.heads
        x = self.input_dropout(x)

        queries = self.to_queries(x)
        queries = self.norm(queries)
        queries = self.query_dropout(queries)

        queries = queries.chunk(2, dim=-1)
        queries = torch.stack(queries).reshape(2, t, b, h, -1)

        dots = torch.einsum("ptbhd,hnpd->tbhpn", queries, self.keys)
        scores, indices = dots.topk(k=self.topk, dim=-1)
        scores, indices = map(lambda x: x.chunk(2, dim=3), (scores, indices))

        all_topk = self.topk ** 2
        shape = (t, b, h, all_topk)

        all_scores = (scores[0][..., :, None] + scores[1][..., None, :]).reshape(*shape)

        all_indices = (
            indices[0][..., :, None] * self.num_keys + indices[1][..., None, :]
        ).reshape(*shape)

        final_topk, final_indices = all_scores.topk(self.topk, dim=-1)
        value_indices = all_indices.gather(-1, final_indices)

        attn = final_topk.softmax(dim=-1)

        value_indices, attn = map(
            lambda x: x.reshape(-1, self.topk * h), (value_indices, attn)
        )

        out = self.values(value_indices, per_sample_weights=attn)
        out = self.value_dropout(out)
        return out.reshape(t, b, e)
