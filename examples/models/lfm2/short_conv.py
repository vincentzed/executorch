from __future__ import annotations

import torch
from executorch.examples.models.llama.attention import ForwardOptions
from executorch.examples.models.llama.feed_forward import FeedForward
from executorch.examples.models.llama.norm import RMSNorm
from torch import nn


class ShortConv(nn.Module):
    """Depthwise short convolution with explicit state IO.

    Conv state is passed in and returned rather than stored as a mutable
    buffer, making the module compatible with AOTI tracing.
    """

    def __init__(self, dim: int, L_cache: int = 3, *, bias: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.L_cache = L_cache

        self.conv = nn.Conv1d(dim, dim, kernel_size=L_cache, padding=0, groups=dim, bias=bias)
        self.B_proj = nn.Linear(dim, dim, bias=bias)
        self.C_proj = nn.Linear(dim, dim, bias=bias)
        self.x_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def forward(
        self, x: torch.Tensor, conv_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if conv_state is None:
            conv_state = torch.zeros(
                1, self.dim, self.L_cache - 1,
                dtype=self.conv.weight.dtype,
                device=self.conv.weight.device,
            )

        B = self.B_proj(x).transpose(-1, -2)
        C = self.C_proj(x).transpose(-1, -2)
        x = self.x_proj(x).transpose(-1, -2)

        Bx = torch.cat([conv_state, B * x], dim=-1)
        new_conv_state = Bx[..., -(self.L_cache - 1) :]

        # Manual depthwise conv — Triton has no template for nn.Conv1d
        # with groups=dim and dynamic sequence length.
        w = self.conv.weight[:, 0, :]
        conv_out = Bx[..., :-2] * w[:, 0:1] + Bx[..., 1:-1] * w[:, 1:2] + Bx[..., 2:] * w[:, 2:3]

        y = self.out_proj((C * conv_out).transpose(-1, -2).contiguous())
        return y, new_conv_state


class ShortConvBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, norm_eps: float, layer_idx: int = -1) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.conv = ShortConv(dim, L_cache=3, bias=False)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.ffn_norm = RMSNorm(dim, norm_eps)
        self.attention_norm = RMSNorm(dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor | None = None,
        freqs_sin: torch.Tensor | None = None,
        attn_options: ForwardOptions | None = None,
    ) -> tuple[torch.Tensor, dict]:
        conv_state = None
        if attn_options is not None:
            conv_states = attn_options.get("conv_states")
            if conv_states is not None:
                conv_state = conv_states.get(self.layer_idx)

        h, new_conv_state = self.conv(self.attention_norm(x), conv_state)
        h = x + h
        out = h + self.feed_forward(self.ffn_norm(h))

        update: dict = {}
        if attn_options is not None and "conv_states" in attn_options:
            states = dict(attn_options["conv_states"])
            states[self.layer_idx] = new_conv_state
            update["conv_states"] = states
        return out, update

    def reset_cache(self) -> None:
        pass
