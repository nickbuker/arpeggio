"""Loss functions for PANTHER pretraining and fraud fine-tuning.

Pretraining uses in-batch sampled softmax (InfoNCE) for next-token prediction.
Fraud fine-tuning uses binary cross-entropy with optional focal loss weighting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InBatchSampledSoftmaxLoss(nn.Module):
    """In-batch sampled softmax loss for autoregressive next-token prediction.

    For each valid position in the batch, the model predicts the next token
    using contrastive learning against all other tokens in the batch as
    negatives (InfoNCE formulation).

    Parameters
    ----------
    temperature : float
        Softmax temperature. Lower values sharpen the distribution.
    l2_norm : bool
        Whether to L2-normalise embeddings before computing similarities.
    """

    def __init__(self, temperature: float = 0.05, l2_norm: bool = True) -> None:
        super().__init__()
        self.temperature = temperature
        self.l2_norm = l2_norm

    def forward(
        self,
        output_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the in-batch sampled softmax loss.

        Parameters
        ----------
        output_embeddings : torch.Tensor
            Encoder outputs at positions t (the "query"), shape ``[B, N, D]``.
        target_embeddings : torch.Tensor
            Item embeddings of the ground-truth next tokens at positions t+1
            (the "key"), shape ``[B, N, D]``.
        mask : torch.Tensor
            Boolean mask of valid (non-padding) positions, shape ``[B, N]``.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        # Shift: predict position t+1 from output at position t
        q = output_embeddings[:, :-1, :]  # [B, N-1, D]
        k = target_embeddings[:, 1:, :]   # [B, N-1, D]
        valid = mask[:, 1:]                # [B, N-1]

        # Flatten to [M, D] keeping only valid positions
        q_flat = q[valid]  # [M, D]
        k_flat = k[valid]  # [M, D]

        if self.l2_norm:
            q_flat = F.normalize(q_flat, dim=-1)
            k_flat = F.normalize(k_flat, dim=-1)

        # Similarity matrix [M, M]; diagonal = positive pairs
        logits = (q_flat @ k_flat.T) / self.temperature  # [M, M]
        labels = torch.arange(len(q_flat), device=q_flat.device)
        return F.cross_entropy(logits, labels)


class FocalBCELoss(nn.Module):
    """Focal loss for binary classification with class imbalance.

    Focal loss down-weights easy negatives so training focuses on hard
    examples, which is important for the heavily imbalanced fraud task.

    Parameters
    ----------
    gamma : float
        Focusing parameter. ``gamma=0`` reduces to standard BCE.
    pos_weight : float or None
        Weight applied to the positive (fraud) class.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self._pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor
            Raw (pre-sigmoid) predictions, shape ``[B]``.
        targets : torch.Tensor
            Binary labels (0 or 1), shape ``[B]``.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        pw = (
            torch.tensor([self._pos_weight], device=logits.device)
            if self._pos_weight is not None
            else None
        )
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()
