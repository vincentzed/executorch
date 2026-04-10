from typing import Optional

import torch
from executorch.examples.models.llama.attention import ForwardOptions
from executorch.examples.models.llama.feed_forward import FeedForward

from executorch.examples.models.llama.norm import RMSNorm
from torch import nn


class ShortConv(nn.Module):
    def __init__(
        self,
        dim: int,
        L_cache: int = 3,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.dim = dim
        self.L_cache = L_cache
        self.device = device
        self.dtype = dtype
        self.bias = bias

        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=L_cache,
            padding=0,
            groups=dim,
            bias=bias,
        )

        ## better performance in Executorch with separate projections
        self.B_proj = nn.Linear(dim, dim, bias=bias)
        self.C_proj = nn.Linear(dim, dim, bias=bias)
        self.x_proj = nn.Linear(dim, dim, bias=bias)

        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def _make_empty_conv_state(self) -> torch.Tensor:
        """Return a zero-initialised conv state: [1, dim, L_cache - 1]."""
        return torch.zeros(
            1, self.dim, self.L_cache - 1,
            device=self.conv.weight.device,
            dtype=self.conv.weight.dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, dim]
            conv_state: [batch_size, dim, L_cache - 1] or None for fresh state.

        Returns:
            (output, new_conv_state) — caller is responsible for persisting
            the returned state between calls.
        """
        if conv_state is None:
            conv_state = self._make_empty_conv_state()

        B = self.B_proj(x).transpose(-1, -2)
        C = self.C_proj(x).transpose(-1, -2)
        x = self.x_proj(x).transpose(-1, -2)

        Bx = B * x
        Bx = torch.cat([conv_state, Bx], dim=-1)

        new_conv_state = Bx[..., -(self.L_cache - 1):]

        # Manual depthwise conv: Triton has no template for nn.Conv1d with
        # groups=dim and dynamic seq_len.  kernel_size is always 3.
        w = self.conv.weight[:, 0, :]  # [dim, 3]
        conv_out = (
            Bx[..., :-2] * w[:, 0:1]
            + Bx[..., 1:-1] * w[:, 1:2]
            + Bx[..., 2:] * w[:, 2:3]
        )
        y = C * conv_out

        y = y.transpose(-1, -2).contiguous()
        y = self.out_proj(y)
        return y, new_conv_state


class ShortConvBlock(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, norm_eps: float, layer_idx: int = -1
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.L_cache = 3
        self.conv = ShortConv(dim, self.L_cache, bias=False)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.ffn_norm = RMSNorm(dim, norm_eps)
        self.attention_norm = RMSNorm(dim, norm_eps)

    def forward(
        self,
        x,
        freqs_cos=None,
        freqs_sin=None,
        attn_options: Optional[ForwardOptions] = None,
    ):
        conv_state = None
        if attn_options is not None:
            conv_states = attn_options.get("conv_states")
            if conv_states is not None:
                conv_state = conv_states.get(self.layer_idx)

        h, new_conv_state = self.conv.forward(self.attention_norm(x), conv_state)
        h = x + h
        out = h + self.feed_forward(self.ffn_norm(h))

        update = {}
        if attn_options is not None and "conv_states" in attn_options:
            states = dict(attn_options["conv_states"])
            states[self.layer_idx] = new_conv_state
            update["conv_states"] = states
        return out, update

    def reset_cache(self):
        pass
