"""PANTHER model: SASRec-based sequential transaction encoder.

Architecture follows the PANTHER paper (https://arxiv.org/html/2510.10102v2):
  - Composite behavior token embeddings concatenated with per-feature embeddings
  - User/card attribute embeddings blended in as a profile signal
  - Causal multi-head self-attention with pre-layer-norm
  - Point-wise feed-forward network with residual connections
  - FraudHead: Deep & Cross Network binary classifier on top of the encoder
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FFN(nn.Module):
    """Position-wise feed-forward network used inside each transformer block."""

    def __init__(self, emb_dim: int, hidden_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, N, D]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, N, D]``.
        """
        return self.net(x)


class _CrossLayer(nn.Module):
    """Single layer of the Cross Network in DCN."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x0 : torch.Tensor
            Original input ``[B, D]``.
        x : torch.Tensor
            Current layer input ``[B, D]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, D]``.
        """
        return x0 * self.weight(x) + self.bias + x


class FraudHead(nn.Module):
    """Deep & Cross Network fraud classifier.

    Takes the pretrained sequence embedding concatenated with the target
    transaction embedding and outputs a fraud probability.

    Parameters
    ----------
    input_dim : int
        Dimension of the concatenated ``[seq_emb; target_emb]`` vector.
    deep_hidden_dim : int
        Hidden dimension of the deep sub-network.
    num_cross_layers : int
        Number of cross layers.
    dropout_rate : float
        Dropout applied in the deep sub-network.
    """

    def __init__(
        self,
        input_dim: int,
        deep_hidden_dim: int = 128,
        num_cross_layers: int = 2,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.cross_layers = nn.ModuleList(
            [_CrossLayer(input_dim) for _ in range(num_cross_layers)]
        )
        self.deep = nn.Sequential(
            nn.Linear(input_dim, deep_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(deep_hidden_dim, deep_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        combine_dim = input_dim + deep_hidden_dim // 2
        self.output = nn.Sequential(
            nn.Linear(combine_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, D]``.

        Returns
        -------
        torch.Tensor
            Fraud probabilities, shape ``[B]``.
        """
        cross = x
        for layer in self.cross_layers:
            cross = layer(x, cross)
        deep = self.deep(x)
        return torch.sigmoid(self.output(torch.cat([cross, deep], dim=-1))).squeeze(-1)


class PANTHERModel(nn.Module):
    """SASRec-based sequential transaction encoder (PANTHER).

    Parameters
    ----------
    token_vocab_size : int
        Number of composite behavior tokens (output of
        ``build_behavior_tokens``).
    item_emb_dim : int
        Embedding dimension for behavior tokens.
    feature_vocab_sizes : dict
        Mapping of feature name to vocabulary size for per-transaction
        features (e.g. ``{"Amount_bin": 100, "MCC_id": 456, ...}``).
    feature_emb_dims : dict
        Embedding dimension for each per-transaction feature.
    user_attr_vocab_sizes : dict
        Vocabulary size for each per-card user attribute.
    user_attr_emb_dims : dict
        Embedding dimension for each user attribute.
    num_blocks : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads. Must divide ``emb_dim`` evenly.
    dropout_rate : float
        Dropout applied throughout.
    max_seq_len : int
        Maximum sequence length (used to register the causal mask).
    """

    def __init__(
        self,
        token_vocab_size: int,
        item_emb_dim: int = 50,
        feature_vocab_sizes: dict[str, int] | None = None,
        feature_emb_dims: dict[str, int] | None = None,
        user_attr_vocab_sizes: dict[str, int] | None = None,
        user_attr_emb_dims: dict[str, int] | None = None,
        num_blocks: int = 4,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()

        feature_vocab_sizes = feature_vocab_sizes or {}
        feature_emb_dims = feature_emb_dims or {k: 8 for k in feature_vocab_sizes}
        user_attr_vocab_sizes = user_attr_vocab_sizes or {}
        user_attr_emb_dims = user_attr_emb_dims or {k: 4 for k in user_attr_vocab_sizes}

        self._item_emb_dim = item_emb_dim
        self._feat_keys = list(feature_vocab_sizes.keys())
        self._attr_keys = list(user_attr_vocab_sizes.keys())

        # Behavior token embedding
        self.item_emb = nn.Embedding(token_vocab_size + 1, item_emb_dim, padding_idx=0)

        # Per-transaction feature embeddings
        self.feature_embs = nn.ModuleDict({
            k: nn.Embedding(feature_vocab_sizes[k] + 1, feature_emb_dims[k], padding_idx=0)
            for k in self._feat_keys
        })

        # Per-card user attribute embeddings
        self.user_attr_embs = nn.ModuleDict({
            k: nn.Embedding(user_attr_vocab_sizes[k] + 1, user_attr_emb_dims[k], padding_idx=0)
            for k in self._attr_keys
        })

        feat_total_dim = sum(feature_emb_dims[k] for k in self._feat_keys)
        self.emb_dim = item_emb_dim + feat_total_dim

        # Project summed user attr embeddings to emb_dim for blending
        attr_total_dim = sum(user_attr_emb_dims[k] for k in self._attr_keys)
        if self._attr_keys:
            self.user_attr_proj = nn.Linear(attr_total_dim, self.emb_dim)

        self.input_dropout = nn.Dropout(dropout_rate)

        # Learnable positional embeddings
        self.pos_emb = nn.Embedding(max_seq_len + 1, self.emb_dim, padding_idx=0)
        nn.init.trunc_normal_(self.pos_emb.weight, std=math.sqrt(1.0 / self.emb_dim))

        # Transformer blocks (pre-LN SASRec style)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.emb_dim,
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=True,
            )
            for _ in range(num_blocks)
        ])
        ffn_hidden_dim = max(self.emb_dim * 4, 256)
        self.ffn_layers = nn.ModuleList([
            _FFN(self.emb_dim, ffn_hidden_dim, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.ln_attn = nn.ModuleList([nn.LayerNorm(self.emb_dim) for _ in range(num_blocks)])
        self.ln_ffn = nn.ModuleList([nn.LayerNorm(self.emb_dim) for _ in range(num_blocks)])
        self.output_ln = nn.LayerNorm(self.emb_dim)

        # Causal mask: True = ignore this position (upper triangle = future)
        causal = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("_causal_mask", causal)

    def get_item_embeddings(
        self,
        past_ids: torch.Tensor,
        features: dict[str, torch.Tensor],
        user_attrs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build combined input embeddings from token IDs and features.

        Parameters
        ----------
        past_ids : torch.Tensor
            Shape ``[B, N]`` — composite behavior token IDs.
        features : dict
            Per-transaction feature tensors, each ``[B, N]``.
        user_attrs : dict
            Per-card attribute tensors, each ``[B]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, N, emb_dim]``.
        """
        B, N = past_ids.shape
        emb = self.item_emb(past_ids) * math.sqrt(self._item_emb_dim)

        feat_parts = [emb]
        for k in self._feat_keys:
            if k in features:
                feat_parts.append(self.feature_embs[k](features[k]))
        combined = torch.cat(feat_parts, dim=-1)  # [B, N, emb_dim]

        # Positional embeddings
        positions = torch.arange(1, N + 1, device=past_ids.device).unsqueeze(0)  # [1, N]
        combined = combined + self.pos_emb(positions)
        combined = self.input_dropout(combined)

        # Blend user profile (0.1 weight) into every position
        if self._attr_keys:
            attr_embs = torch.cat(
                [self.user_attr_embs[k](user_attrs[k]) for k in self._attr_keys], dim=-1
            )  # [B, attr_total_dim]
            profile = self.user_attr_proj(attr_embs)  # [B, emb_dim]
            combined = combined * 0.9 + profile.unsqueeze(1) * 0.1

        return combined

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Run the transformer stack.

        Parameters
        ----------
        past_lengths : torch.Tensor
            Shape ``[B]`` — actual (unpadded) sequence lengths.
        past_ids : torch.Tensor
            Shape ``[B, N]`` — used to derive the padding mask.
        past_embeddings : torch.Tensor
            Shape ``[B, N, D]`` — output of :meth:`get_item_embeddings`.

        Returns
        -------
        torch.Tensor
            Contextual embeddings, shape ``[B, N, D]``.
        """
        B, N, D = past_embeddings.shape
        x = past_embeddings

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        causal = self._causal_mask[:N, :N]  # [N, N]

        for attn, ffn, ln_a, ln_f in zip(
            self.attention_layers, self.ffn_layers, self.ln_attn, self.ln_ffn
        ):
            normed = ln_a(x)
            attn_out, _ = attn(normed, normed, normed, attn_mask=causal)
            x = x + attn_out
            x = x + ffn(ln_f(x))
            x = x * valid_mask

        return self.output_ln(x)

    def get_sequence_embedding(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        features: dict[str, torch.Tensor],
        user_attrs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return the hidden state at the last valid position for each sample.

        Parameters
        ----------
        past_lengths : torch.Tensor
            Shape ``[B]``.
        past_ids : torch.Tensor
            Shape ``[B, N]``.
        features : dict
            Per-transaction feature tensors, each ``[B, N]``.
        user_attrs : dict
            Per-card attribute tensors, each ``[B]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, D]``.
        """
        embs = self.get_item_embeddings(past_ids, features, user_attrs)
        out = self.forward(past_lengths, past_ids, embs)
        # Index the last valid position per sample
        # past_ids is left-padded, so last valid = position N-1
        idx = (past_lengths - 1).clamp(min=0)  # [B]
        # Gather: out[:, -1, :] works because padding is on the left
        return out[torch.arange(out.size(0), device=out.device), past_lengths - 1]


def build_model_from_artifacts(
    artifacts: dict,
    item_emb_dim: int = 50,
    feature_emb_dim: int = 8,
    user_attr_emb_dim: int = 4,
    num_blocks: int = 4,
    num_heads: int = 2,
    dropout_rate: float = 0.2,
    max_seq_len: int = 512,
) -> PANTHERModel:
    """Construct a :class:`PANTHERModel` from preprocessing artifacts.

    Parameters
    ----------
    artifacts : dict
        The ``artifacts`` dict returned by ``src.preprocessing.preprocess``.
    item_emb_dim : int
        Embedding dim for composite behavior tokens.
    feature_emb_dim : int
        Embedding dim for each per-transaction feature.
    user_attr_emb_dim : int
        Embedding dim for each per-card user attribute.
    num_blocks : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    dropout_rate : float
        Dropout rate throughout.
    max_seq_len : int
        Maximum sequence length.

    Returns
    -------
    PANTHERModel
    """
    token_vocab_size = len(artifacts["token_vocab"])
    vocabs = artifacts["vocabs"]

    feat_cols = artifacts.get("beh_feature_seq_columns", [])
    bin_edges = artifacts.get("bin_edges", {})
    feature_vocab_sizes: dict[str, int] = {}
    for c in feat_cols:
        if c.endswith("_bin"):
            # Binned column: vocab size equals the number of bin edges
            base = c[:-4]
            feature_vocab_sizes[c] = len(bin_edges[base]) if base in bin_edges else 100
        elif c.endswith("_id"):
            # Encoded categorical: look up original column name in vocabs
            base = c[:-3]
            feature_vocab_sizes[c] = len(vocabs[base]) if base in vocabs else 100
        else:
            feature_vocab_sizes[c] = 100
    feature_emb_dims = {c: feature_emb_dim for c in feat_cols}

    attr_cols = artifacts.get("user_attr_id_columns", [])
    # Strip the _id suffix to look up in vocabs
    user_attr_vocab_sizes = {}
    for col in attr_cols:
        base = col[:-3]  # remove "_id"
        if base in vocabs:
            user_attr_vocab_sizes[col] = len(vocabs[base])
    user_attr_emb_dims = {col: user_attr_emb_dim for col in user_attr_vocab_sizes}

    return PANTHERModel(
        token_vocab_size=token_vocab_size,
        item_emb_dim=item_emb_dim,
        feature_vocab_sizes=feature_vocab_sizes,
        feature_emb_dims=feature_emb_dims,
        user_attr_vocab_sizes=user_attr_vocab_sizes,
        user_attr_emb_dims=user_attr_emb_dims,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        max_seq_len=max_seq_len,
    )
