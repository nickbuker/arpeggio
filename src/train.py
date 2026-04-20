"""Training and evaluation loops for PANTHER pretraining and fraud detection.

Pretraining:
  Self-supervised next-token prediction with in-batch sampled softmax loss,
  AdamW optimiser, and a warmup-stable-decay learning rate schedule.

Fraud fine-tuning:
  Freeze (or optionally fine-tune) the pretrained encoder; train a Deep &
  Cross Network head with focal BCE loss on balanced fraud batches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.losses import FocalBCELoss, InBatchSampledSoftmaxLoss
from src.model import FraudHead, PANTHERModel


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def get_wsd_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 1_000,
    stable_steps: int = 8_000,
    decay_steps: int = 1_000,
) -> SequentialLR:
    """Build a Warmup-Stable-Decay learning rate schedule.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimiser whose LR will be scheduled.
    warmup_steps : int
        Number of linear warm-up steps from near-zero to the base LR.
    stable_steps : int
        Number of steps at the constant base LR.
    decay_steps : int
        Number of steps to linearly decay from base LR to 10% of base LR.

    Returns
    -------
    SequentialLR
    """
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps),
            ConstantLR(optimizer, factor=1.0, total_iters=stable_steps),
            LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=decay_steps),
            ConstantLR(optimizer, factor=0.1, total_iters=10 ** 9),
        ],
        milestones=[warmup_steps, warmup_steps + stable_steps, warmup_steps + stable_steps + decay_steps],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_device(batch: dict, device: torch.device) -> dict:
    """Recursively move all tensors in a nested dict batch to *device*."""
    out: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _to_device(v, device)
        else:
            out[k] = v
    return out


def _fraud_sampler(fraud_labels: list[int]) -> WeightedRandomSampler:
    """Build a weighted sampler that up-samples fraud transactions 5×."""
    labels = torch.tensor(fraud_labels)
    n_fraud = labels.sum().item()
    n_normal = len(labels) - n_fraud
    weight_fraud = n_normal / max(n_fraud, 1)
    weights = torch.where(labels == 1, torch.tensor(weight_fraud), torch.tensor(1.0))
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------


def pretrain(
    model: PANTHERModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    warmup_steps: int = 1_000,
    stable_steps: int = 8_000,
    decay_steps: int = 1_000,
    grad_clip: float = 0.1,
    log_interval: int = 100,
    checkpoint_dir: str | Path | None = None,
    device: torch.device | None = None,
) -> list[float]:
    """Pretrain the PANTHER encoder with next-token prediction.

    Parameters
    ----------
    model : PANTHERModel
        Uninitialised or randomly initialised model.
    train_loader : DataLoader
        Yields batches from :class:`src.dataset.TransactionDataset`.
    val_loader : DataLoader, optional
        If provided, validation loss is logged at the end of each epoch.
    num_epochs : int
        Total training epochs.
    learning_rate : float
        Peak learning rate for AdamW.
    weight_decay : float
        L2 regularisation coefficient.
    warmup_steps : int
        Linear warm-up steps.
    stable_steps : int
        Constant-LR steps after warm-up.
    decay_steps : int
        Linear decay steps.
    grad_clip : float
        Gradient norm clipping threshold.
    log_interval : int
        Print training loss every this many steps.
    checkpoint_dir : str or Path, optional
        If provided, the best model checkpoint is saved here.
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.

    Returns
    -------
    list of float
        Per-epoch training loss.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = InBatchSampledSoftmaxLoss(temperature=0.05, l2_norm=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.98), weight_decay=weight_decay
    )
    scheduler = get_wsd_scheduler(optimizer, warmup_steps, stable_steps, decay_steps)

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epoch_losses: list[float] = []
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(num_epochs):
        print(f"[pretrain] epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = _to_device(batch, device)

            past_ids = batch["past_ids"]
            past_lengths = batch["past_lengths"]
            features = batch["features"]
            user_attrs = batch["user_attrs"]

            embeddings = model.get_item_embeddings(past_ids, features, user_attrs)
            output = model.forward(past_lengths, past_ids, embeddings)

            valid_mask = past_ids != 0  # [B, N]
            loss = criterion(output, embeddings, valid_mask)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % log_interval == 0:
                avg = running_loss / n_batches
                print(f"[pretrain] epoch {epoch+1} step {global_step} loss={avg:.4f}")

        epoch_loss = running_loss / max(n_batches, 1)
        epoch_losses.append(epoch_loss)

        if val_loader is not None:
            val_loss = _eval_pretrain_loss(model, val_loader, criterion, device)
            print(f"[pretrain] epoch {epoch+1} train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}")
            if checkpoint_dir is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), checkpoint_dir / "panther_best.pt")
        else:
            print(f"[pretrain] epoch {epoch+1} train_loss={epoch_loss:.4f}")
            if checkpoint_dir is not None:
                torch.save(model.state_dict(), checkpoint_dir / f"panther_epoch{epoch+1}.pt")

    return epoch_losses


def _eval_pretrain_loss(
    model: PANTHERModel,
    loader: DataLoader,
    criterion: InBatchSampledSoftmaxLoss,
    device: torch.device,
) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            embs = model.get_item_embeddings(
                batch["past_ids"], batch["features"], batch["user_attrs"]
            )
            out = model.forward(batch["past_lengths"], batch["past_ids"], embs)
            mask = batch["past_ids"] != 0
            total += criterion(out, embs, mask).item()
            n += 1
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Fraud fine-tuning
# ---------------------------------------------------------------------------


def finetune_fraud(
    encoder: PANTHERModel,
    fraud_head: FraudHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 5,
    learning_rate: float = 1e-4,
    freeze_encoder: bool = True,
    focal_gamma: float = 2.0,
    grad_clip: float = 0.1,
    log_interval: int = 50,
    checkpoint_dir: str | Path | None = None,
    device: torch.device | None = None,
) -> list[dict[str, float]]:
    """Fine-tune a fraud detection head on top of a pretrained encoder.

    Parameters
    ----------
    encoder : PANTHERModel
        Pretrained encoder; weights may be frozen.
    fraud_head : FraudHead
        Classification head to train.
    train_loader : DataLoader
        Yields batches from :class:`src.dataset.FraudDataset`.
    val_loader : DataLoader
        Validation batches for early stopping and metric logging.
    num_epochs : int
        Training epochs.
    learning_rate : float
        Learning rate for AdamW.
    freeze_encoder : bool
        If ``True``, only the fraud head parameters are updated.
    focal_gamma : float
        Focal loss gamma. Set to ``0`` for standard BCE.
    grad_clip : float
        Gradient norm clipping threshold.
    log_interval : int
        Print training loss every this many steps.
    checkpoint_dir : str or Path, optional
        Best model checkpoint destination.
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.

    Returns
    -------
    list of dict
        Per-epoch metrics dicts with keys ``train_loss``, ``val_auc``,
        ``val_f1``, ``val_precision``, ``val_recall``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = encoder.to(device)
    fraud_head = fraud_head.to(device)

    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder.eval()

    criterion = FocalBCELoss(gamma=focal_gamma)
    params = (
        fraud_head.parameters()
        if freeze_encoder
        else list(encoder.parameters()) + list(fraud_head.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=learning_rate)

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    best_auc = 0.0
    global_step = 0

    for epoch in range(num_epochs):
        print(f"[fraud] epoch {epoch+1}/{num_epochs}")
        fraud_head.train()
        if not freeze_encoder:
            encoder.train()

        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = _to_device(batch, device)

            seq_emb = encoder.get_sequence_embedding(
                batch["past_lengths"],
                batch["past_ids"],
                batch["features"],
                batch["user_attrs"],
            )
            target_emb = encoder.item_emb(batch["target_id"])
            head_input = torch.cat([seq_emb, target_emb], dim=-1)

            probs = fraud_head(head_input)
            # FocalBCELoss expects logits; convert from sigmoid output back to logit
            logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
            loss = criterion(logits, batch["fraud_label"])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(fraud_head.parameters()) + ([] if freeze_encoder else list(encoder.parameters())),
                grad_clip,
            )
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % log_interval == 0:
                print(
                    f"[fraud] epoch {epoch+1} step {global_step} "
                    f"loss={running_loss/n_batches:.4f}"
                )

        train_loss = running_loss / max(n_batches, 1)
        metrics = evaluate_fraud(encoder, fraud_head, val_loader, device)
        metrics["train_loss"] = train_loss
        history.append(metrics)

        print(
            f"[fraud] epoch {epoch+1} train_loss={train_loss:.4f} "
            f"val_auc={metrics['auc']:.4f} val_f1={metrics['f1']:.4f}"
        )

        if checkpoint_dir is not None and metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(
                {"encoder": encoder.state_dict(), "fraud_head": fraud_head.state_dict()},
                checkpoint_dir / "fraud_best.pt",
            )

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_fraud(
    encoder: PANTHERModel,
    fraud_head: FraudHead,
    loader: DataLoader,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Evaluate the fraud detection model.

    Parameters
    ----------
    encoder : PANTHERModel
        Pretrained (and optionally fine-tuned) encoder.
    fraud_head : FraudHead
        Trained classification head.
    loader : DataLoader
        Yields batches from :class:`src.dataset.FraudDataset`.
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.

    Returns
    -------
    dict
        Metrics: ``auc``, ``average_precision``, ``f1``, ``precision``,
        ``recall``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder.eval()
    fraud_head.eval()

    all_probs: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            seq_emb = encoder.get_sequence_embedding(
                batch["past_lengths"],
                batch["past_ids"],
                batch["features"],
                batch["user_attrs"],
            )
            target_emb = encoder.item_emb(batch["target_id"])
            probs = fraud_head(torch.cat([seq_emb, target_emb], dim=-1))
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(batch["fraud_label"].cpu().int().tolist())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    return {
        "auc": roc_auc_score(all_labels, all_probs),
        "average_precision": average_precision_score(all_labels, all_probs),
        "f1": f1_score(all_labels, preds, zero_division=0),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall": recall_score(all_labels, preds, zero_division=0),
    }


def evaluate_retrieval(
    model: PANTHERModel,
    loader: DataLoader,
    top_k: tuple[int, ...] = (1, 10, 100),
    device: torch.device | None = None,
) -> dict[str, float]:
    """Evaluate next-token retrieval (HR@K and NDCG@K).

    For each sample the last token of the sequence is held out as the target.
    The model predicts a ranking over the full vocabulary by dot-product
    similarity between the encoder output and all item embeddings.

    Parameters
    ----------
    model : PANTHERModel
        Pretrained encoder.
    loader : DataLoader
        Yields batches from :class:`src.dataset.TransactionDataset`.
    top_k : tuple of int
        K values at which to compute HR and NDCG.
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.

    Returns
    -------
    dict
        Metrics ``hr@K`` and ``ndcg@K`` for each K in *top_k*.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    # Build the full item embedding matrix once
    vocab_size = model.item_emb.num_embeddings
    all_item_ids = torch.arange(vocab_size, device=device)
    all_item_embs = model.item_emb(all_item_ids)  # [V, D]

    hits: dict[int, int] = {k: 0 for k in top_k}
    ndcg: dict[int, float] = {k: 0.0 for k in top_k}
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            past_ids = batch["past_ids"]
            past_lengths = batch["past_lengths"]

            # Hold out the last token: shift sequence left by 1
            history_ids = past_ids.clone()
            target_ids = past_ids[
                torch.arange(past_ids.size(0)), past_lengths - 1
            ]  # [B]
            # Zero out the last valid token so it's not attended to
            history_ids[torch.arange(past_ids.size(0)), past_lengths - 1] = 0
            history_lengths = (past_lengths - 1).clamp(min=1)

            seq_emb = model.get_sequence_embedding(
                history_lengths, history_ids, batch["features"], batch["user_attrs"]
            )  # [B, D]

            scores = seq_emb @ all_item_embs.T  # [B, V]
            top_indices = scores.topk(max(top_k), dim=-1).indices  # [B, K_max]

            for b in range(past_ids.size(0)):
                target = target_ids[b].item()
                for k in top_k:
                    ranked = top_indices[b, :k].tolist()
                    if target in ranked:
                        hits[k] += 1
                        rank = ranked.index(target) + 1
                        import math
                        ndcg[k] += 1.0 / math.log2(rank + 1)
            n_samples += past_ids.size(0)

    return {
        **{f"hr@{k}": hits[k] / max(n_samples, 1) for k in top_k},
        **{f"ndcg@{k}": ndcg[k] / max(n_samples, 1) for k in top_k},
    }
