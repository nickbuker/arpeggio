"""PyTorch datasets for PANTHER pretraining and fraud fine-tuning."""

from __future__ import annotations

import polars as pl
import torch
from torch.utils.data import Dataset


class TransactionDataset(Dataset):
    """Sequence dataset for next-token pretraining.

    Each item is a padded/truncated transaction history ready for
    autoregressive next-token prediction.

    Parameters
    ----------
    sequences_df : pl.DataFrame
        Output of ``src.preprocessing.assemble_sequences``.
    max_length : int
        Sequences longer than this are truncated to the most recent tokens.
    """

    def __init__(self, sequences_df: pl.DataFrame, max_length: int = 512) -> None:
        self._df = sequences_df
        self._max_length = max_length
        self._feat_cols = [
            c for c in ["Amount_bin_seq", "MCC_id_seq", "Use Chip_id_seq"]
            if c in sequences_df.columns
        ]
        self._attr_cols = [
            c for c in sequences_df.columns
            if c.endswith("_id") and not c.endswith("_seq") and c != "Card"
        ]

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict:
        row = self._df.row(idx, named=True)
        seq = row["beh_seq"][-self._max_length:]
        length = len(seq)
        pad = self._max_length - length
        padded_seq = [0] * pad + seq

        item: dict = {
            "past_ids": torch.tensor(padded_seq, dtype=torch.long),
            "past_lengths": torch.tensor(length, dtype=torch.long),
            "features": {},
            "user_attrs": {},
        }

        for col in self._feat_cols:
            feat = row[col][-self._max_length:]
            feat_pad = self._max_length - len(feat)
            item["features"][col.replace("_seq", "")] = torch.tensor(
                [0] * feat_pad + list(feat), dtype=torch.long
            )

        for col in self._attr_cols:
            item["user_attrs"][col] = torch.tensor(int(row[col]), dtype=torch.long)

        return item


class FraudDataset(Dataset):
    """Dataset for fraud detection fine-tuning.

    Each item uses all but the last transaction as context and predicts
    whether the last transaction is fraudulent.

    Parameters
    ----------
    sequences_df : pl.DataFrame
        Output of ``src.preprocessing.assemble_sequences`` (with fraud
        labels; i.e., built without ``pretrain=True``).
    max_history_length : int
        Maximum number of context transactions to keep per sample.
    """

    def __init__(
        self,
        sequences_df: pl.DataFrame,
        max_history_length: int = 512,
    ) -> None:
        self._df = sequences_df
        self._max_history = max_history_length
        self._feat_cols = [
            c for c in ["Amount_bin_seq", "MCC_id_seq", "Use Chip_id_seq"]
            if c in sequences_df.columns
        ]
        self._attr_cols = [
            c for c in sequences_df.columns
            if c.endswith("_id") and not c.endswith("_seq") and c != "Card"
        ]

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict:
        row = self._df.row(idx, named=True)

        history = row["beh_seq"][:-1][-self._max_history:]
        target_token = row["beh_seq"][-1]
        fraud_label = 1 if row["fraud_labels"][-1] == "Yes" else 0

        length = len(history)
        pad = self._max_history - length
        padded_history = [0] * pad + list(history)

        item: dict = {
            "past_ids": torch.tensor(padded_history, dtype=torch.long),
            "past_lengths": torch.tensor(length, dtype=torch.long),
            "target_id": torch.tensor(target_token, dtype=torch.long),
            "fraud_label": torch.tensor(fraud_label, dtype=torch.float),
            "features": {},
            "user_attrs": {},
        }

        for col in self._feat_cols:
            feat = row[col][:-1][-self._max_history:]
            feat_pad = self._max_history - len(feat)
            item["features"][col.replace("_seq", "")] = torch.tensor(
                [0] * feat_pad + list(feat), dtype=torch.long
            )

        for col in self._attr_cols:
            item["user_attrs"][col] = torch.tensor(int(row[col]), dtype=torch.long)

        return item
