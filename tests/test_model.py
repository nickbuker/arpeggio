"""Unit tests for model, dataset, and loss components."""

import torch
import pandas as pd
import pytest

from src.dataset import FraudDataset, TransactionDataset
from src.losses import FocalBCELoss, InBatchSampledSoftmaxLoss
from src.model import FraudHead, PANTHERModel, build_model_from_artifacts


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 200
ITEM_EMB_DIM = 16
MAX_SEQ = 8
BATCH = 4
FEAT_VOCAB = {"Amount_bin": 10, "MCC_id": 20, "Use Chip_id": 5}
FEAT_DIMS = {"Amount_bin": 4, "MCC_id": 4, "Use Chip_id": 4}
ATTR_VOCAB = {"Card Brand_id": 5, "Card Type_id": 3}
ATTR_DIMS = {"Card Brand_id": 2, "Card Type_id": 2}


@pytest.fixture
def model() -> PANTHERModel:
    return PANTHERModel(
        token_vocab_size=VOCAB_SIZE,
        item_emb_dim=ITEM_EMB_DIM,
        feature_vocab_sizes=FEAT_VOCAB,
        feature_emb_dims=FEAT_DIMS,
        user_attr_vocab_sizes=ATTR_VOCAB,
        user_attr_emb_dims=ATTR_DIMS,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.0,
        max_seq_len=MAX_SEQ,
    )


@pytest.fixture
def batch() -> dict:
    """Minimal batch dict matching TransactionDataset output."""
    lengths = torch.tensor([5, 3, MAX_SEQ, 6])
    ids = torch.zeros(BATCH, MAX_SEQ, dtype=torch.long)
    for i, l in enumerate(lengths):
        ids[i, -l:] = torch.randint(1, VOCAB_SIZE, (l,))
    return {
        "past_ids": ids,
        "past_lengths": lengths,
        "features": {
            k: torch.where(ids != 0, torch.randint(1, v, (BATCH, MAX_SEQ)), torch.zeros(BATCH, MAX_SEQ, dtype=torch.long))
            for k, v in FEAT_VOCAB.items()
        },
        "user_attrs": {
            "Card Brand_id": torch.randint(1, 5, (BATCH,)),
            "Card Type_id": torch.randint(1, 3, (BATCH,)),
        },
    }


# ---------------------------------------------------------------------------
# PANTHERModel
# ---------------------------------------------------------------------------


def test_model_emb_dim(model: PANTHERModel) -> None:
    expected = ITEM_EMB_DIM + sum(FEAT_DIMS.values())
    assert model.emb_dim == expected


def test_get_item_embeddings_shape(model: PANTHERModel, batch: dict) -> None:
    embs = model.get_item_embeddings(
        batch["past_ids"], batch["features"], batch["user_attrs"]
    )
    assert embs.shape == (BATCH, MAX_SEQ, model.emb_dim)


def test_forward_output_shape(model: PANTHERModel, batch: dict) -> None:
    embs = model.get_item_embeddings(
        batch["past_ids"], batch["features"], batch["user_attrs"]
    )
    out = model.forward(batch["past_lengths"], batch["past_ids"], embs)
    assert out.shape == (BATCH, MAX_SEQ, model.emb_dim)


def test_forward_pads_zeroed(model: PANTHERModel, batch: dict) -> None:
    """Padding positions (past_ids == 0) must be zeroed in the output."""
    model.eval()
    with torch.no_grad():
        embs = model.get_item_embeddings(
            batch["past_ids"], batch["features"], batch["user_attrs"]
        )
        out = model.forward(batch["past_lengths"], batch["past_ids"], embs)
    pad_mask = batch["past_ids"] == 0  # [B, N]
    assert out[pad_mask].abs().max().item() == pytest.approx(0.0, abs=1e-5)


def test_get_sequence_embedding_shape(model: PANTHERModel, batch: dict) -> None:
    model.eval()
    with torch.no_grad():
        seq_emb = model.get_sequence_embedding(
            batch["past_lengths"],
            batch["past_ids"],
            batch["features"],
            batch["user_attrs"],
        )
    assert seq_emb.shape == (BATCH, model.emb_dim)


def test_model_no_attr(batch: dict) -> None:
    """Model without user attrs should still produce correct output shapes."""
    m = PANTHERModel(
        token_vocab_size=VOCAB_SIZE,
        item_emb_dim=ITEM_EMB_DIM,
        feature_vocab_sizes=FEAT_VOCAB,
        feature_emb_dims=FEAT_DIMS,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.0,
        max_seq_len=MAX_SEQ,
    )
    embs = m.get_item_embeddings(batch["past_ids"], batch["features"], {})
    out = m.forward(batch["past_lengths"], batch["past_ids"], embs)
    assert out.shape == (BATCH, MAX_SEQ, m.emb_dim)


# ---------------------------------------------------------------------------
# FraudHead
# ---------------------------------------------------------------------------


def test_fraud_head_output_shape(model: PANTHERModel, batch: dict) -> None:
    head = FraudHead(input_dim=model.emb_dim + model._item_emb_dim, deep_hidden_dim=32)
    model.eval()
    with torch.no_grad():
        seq_emb = model.get_sequence_embedding(
            batch["past_lengths"], batch["past_ids"], batch["features"], batch["user_attrs"]
        )
        target_emb = model.item_emb(torch.randint(1, VOCAB_SIZE, (BATCH,)))
        out = head(torch.cat([seq_emb, target_emb], dim=-1))
    assert out.shape == (BATCH,)


def test_fraud_head_output_range(model: PANTHERModel, batch: dict) -> None:
    """Fraud probabilities must lie in [0, 1]."""
    head = FraudHead(input_dim=model.emb_dim + model._item_emb_dim, deep_hidden_dim=32)
    model.eval()
    with torch.no_grad():
        seq_emb = model.get_sequence_embedding(
            batch["past_lengths"], batch["past_ids"], batch["features"], batch["user_attrs"]
        )
        target_emb = model.item_emb(torch.randint(1, VOCAB_SIZE, (BATCH,)))
        out = head(torch.cat([seq_emb, target_emb], dim=-1))
    assert (out >= 0).all() and (out <= 1).all()


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def test_in_batch_softmax_loss_scalar() -> None:
    B, N, D = 4, 8, 16
    q = torch.randn(B, N, D)
    k = torch.randn(B, N, D)
    mask = torch.ones(B, N, dtype=torch.bool)
    loss_fn = InBatchSampledSoftmaxLoss()
    loss = loss_fn(q, k, mask)
    assert loss.shape == ()
    assert loss.item() > 0


def test_in_batch_softmax_loss_ignores_padding() -> None:
    B, N, D = 4, 8, 16
    q = torch.randn(B, N, D)
    k = torch.randn(B, N, D)
    # Only first 4 positions valid
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, :4] = True
    loss_fn = InBatchSampledSoftmaxLoss()
    loss = loss_fn(q, k, mask)
    assert loss.item() > 0


def test_focal_bce_loss_scalar() -> None:
    logits = torch.randn(8)
    labels = torch.randint(0, 2, (8,)).float()
    loss_fn = FocalBCELoss(gamma=2.0)
    loss = loss_fn(logits, labels)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_focal_bce_gamma_zero_matches_bce() -> None:
    """gamma=0 focal loss should equal standard BCE."""
    torch.manual_seed(0)
    logits = torch.randn(16)
    labels = torch.randint(0, 2, (16,)).float()
    focal = FocalBCELoss(gamma=0.0)(logits, labels).item()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    assert focal == pytest.approx(bce, rel=1e-4)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _make_sequences_df(n_cards: int = 10, seq_len: int = 40) -> pd.DataFrame:
    rows = []
    for i in range(n_cards):
        seq = list(torch.randint(1, 200, (seq_len,)).numpy())
        rows.append(
            {
                "User": i,
                "Card": 0,
                "beh_seq": seq,
                "timestamps": [None] * seq_len,
                "fraud_labels": (["No"] * (seq_len - 1)) + ["Yes"],
                "seq_length": seq_len,
                "Amount_bin_seq": list(torch.randint(1, 10, (seq_len,)).numpy()),
                "MCC_id_seq": list(torch.randint(1, 20, (seq_len,)).numpy()),
                "Use Chip_id_seq": list(torch.randint(1, 5, (seq_len,)).numpy()),
                "Card Brand_id": 1,
                "Card Type_id": 2,
            }
        )
    return pd.DataFrame(rows)


def test_transaction_dataset_length() -> None:
    df = _make_sequences_df(n_cards=5)
    ds = TransactionDataset(df, max_length=32)
    assert len(ds) == 5


def test_transaction_dataset_shapes() -> None:
    df = _make_sequences_df(n_cards=4)
    ds = TransactionDataset(df, max_length=32)
    item = ds[0]
    assert item["past_ids"].shape == (32,)
    assert item["past_lengths"].item() <= 32
    for v in item["features"].values():
        assert v.shape == (32,)


def test_transaction_dataset_left_padded() -> None:
    """Sequences shorter than max_length must be left-padded with zeros."""
    df = _make_sequences_df(n_cards=4, seq_len=10)
    ds = TransactionDataset(df, max_length=32)
    item = ds[0]
    assert (item["past_ids"][:22] == 0).all()
    assert (item["past_ids"][22:] != 0).any()


def test_fraud_dataset_length() -> None:
    df = _make_sequences_df(n_cards=6)
    ds = FraudDataset(df, max_history_length=32)
    assert len(ds) == 6


def test_fraud_dataset_label_is_binary() -> None:
    df = _make_sequences_df(n_cards=6)
    ds = FraudDataset(df)
    for i in range(len(ds)):
        label = ds[i]["fraud_label"].item()
        assert label in (0.0, 1.0)


def test_fraud_dataset_last_fraud_label() -> None:
    """Last transaction marked 'Yes' should yield label=1."""
    df = _make_sequences_df(n_cards=4)
    ds = FraudDataset(df)
    for i in range(len(ds)):
        assert ds[i]["fraud_label"].item() == 1.0


def test_fraud_dataset_history_excludes_target() -> None:
    """History length should be seq_length - 1."""
    df = _make_sequences_df(n_cards=4, seq_len=10)
    ds = FraudDataset(df, max_history_length=64)
    item = ds[0]
    assert item["past_lengths"].item() == 9


# ---------------------------------------------------------------------------
# build_model_from_artifacts
# ---------------------------------------------------------------------------


def test_build_model_from_artifacts() -> None:
    artifacts = {
        "token_vocab": {str(i): i for i in range(1, 101)},
        "vocabs": {
            "Amount_bin": {str(i): i for i in range(1, 101)},
            "MCC": {str(i): i for i in range(1, 51)},
            "Use Chip": {"Swipe Transaction": 1, "Online Transaction": 2},
            "Card Brand": {"Visa": 1, "Mastercard": 2},
            "Card Type": {"Debit": 1, "Credit": 2},
        },
        "beh_feature_seq_columns": ["Amount_bin", "MCC_id", "Use Chip_id"],
        "user_attr_id_columns": ["Card Brand_id", "Card Type_id"],
    }
    m = build_model_from_artifacts(artifacts, item_emb_dim=16, max_seq_len=8)
    assert isinstance(m, PANTHERModel)
    assert m.item_emb.num_embeddings == 101  # vocab_size + 1
