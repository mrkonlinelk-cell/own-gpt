"""
A decoder-only Transformer language model, written from scratch in PyTorch.

This is the same family of architecture as GPT. It reads a sequence of token
ids and, for every position, predicts the next token.

The important pieces, in plain language:

- Token embeddings: each word-piece becomes a vector of numbers.
- Position embeddings: the model also needs to know the order of the tokens.
- Causal self-attention: each token looks at earlier tokens (never later
  ones, because those would be "cheating" when we predict the future).
- MLP (a small feed-forward network): extra thinking capacity after attention.
- LayerNorm and residual connections: these keep training stable.
- The final linear layer (tied to the token embeddings) turns the last hidden
  vector into a score for every word in the vocabulary.

No pretrained weights are loaded here. Training starts from random numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (look-behind-only) mask."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        # One big linear layer builds Query, Key and Value for all heads at once.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _manual_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Explicit matmul attention. Used for ONNX export, where
        scaled_dot_product_attention with is_causal is unreliable.
        q, k, v: (batch, n_head, time, head_dim)
        """
        t = q.size(-2)
        scale = 1.0 / math.sqrt(self.head_dim)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale
        # Causal mask: position i may only attend to positions <= i.
        q_idx = torch.arange(t, device=q.device).unsqueeze(1)
        k_idx = torch.arange(t, device=q.device).unsqueeze(0)
        # Use a large negative number instead of -inf so softmax stays ONNX-friendly.
        att = att.masked_fill(q_idx < k_idx, -1e9)
        att = torch.softmax(att, dim=-1)
        if self.training and self.attn_dropout.p > 0:
            att = self.attn_dropout(att)
        return torch.matmul(att, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, n_embd)
        b, t, c = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # (batch, n_head, time, head_dim)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        if torch.onnx.is_in_onnx_export() or getattr(self, "force_manual_attention", False):
            y = self._manual_attention(q, k, v)
        else:
            # Scaled dot-product attention with an automatic causal mask.
            # is_causal=True means position i may only attend to positions <= i.
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """A two-layer feed-forward network applied at each position."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """One transformer block: attention, then MLP, each with a residual path."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm layout (used by GPT-2 and most modern GPTs).
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: the same numbers that look up a token also score it.
        # This saves parameters and usually helps a small model.
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        # GPT-2 style residual scaling for the projection layers at the end
        # of attention and the MLP. This keeps the residual stream stable
        # when we stack several blocks.
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("c_proj.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx: (batch, time) token ids
        targets: optional (batch, time) next-token ids for the training loss
        returns logits (batch, time, vocab) and optional mean cross-entropy loss
        """
        b, t = idx.size()
        if t > self.config.block_size:
            raise ValueError(f"Sequence length {t} exceeds block_size {self.config.block_size}")

        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
        stop_id: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive sampling: predict one token, append it, repeat.

        temperature < 1 makes the model more confident / less random.
        top_k keeps only the k most likely tokens before sampling.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, k, dim=-1).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            if stop_id is not None and int(next_id.item()) == stop_id:
                break
        return idx

    def count_parameters(self) -> int:
        # Tied weights are stored once, so we count unique parameter tensors.
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total


class GPTLogits(nn.Module):
    """
    Thin wrapper for ONNX export: tokens (batch, seq) -> last-step logits (batch, vocab).
    The browser only needs the next-token scores, not the full sequence of logits.
    """

    def __init__(self, model: GPT) -> None:
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(idx)
        return logits[:, -1, :]
