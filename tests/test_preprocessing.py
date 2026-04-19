"""Unit tests for src.preprocessing."""

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from src.preprocessing import (
    MIN_SEQ_LENGTH,
    add_temporal_features,
    apply_binning,
    assemble_sequences,
    build_behavior_tokens,
    build_datetime,
    build_vocab,
    encode_categoricals,
    log_bin,
    parse_currency_column,
    parse_dollar_amount,
    save_artifacts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_transactions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "User": [0, 0, 0, 1, 1],
            "Card": [0, 0, 0, 0, 0],
            "Year": [2002, 2002, 2003, 2002, 2002],
            "Month": [9, 9, 1, 9, 10],
            "Day": [1, 2, 5, 1, 3],
            "Time": ["06:21", "17:45", "09:00", "08:00", "12:30"],
            "Amount": ["$134.09", "$38.48", "$200.00", "$50.00", "$75.00"],
            "Use Chip": [
                "Swipe Transaction",
                "Online Transaction",
                "Chip Transaction",
                "Swipe Transaction",
                "Online Transaction",
            ],
            "MCC": [5300, 5411, 5651, 5300, 5411],
            "Is Fraud?": ["No", "No", "Yes", "No", "No"],
        }
    )


@pytest.fixture
def sample_transactions_with_datetime(sample_transactions) -> pl.DataFrame:
    return sample_transactions.with_columns(
        parse_dollar_amount(sample_transactions["Amount"]).alias("Amount"),
        build_datetime(sample_transactions).alias("datetime"),
    )


# ---------------------------------------------------------------------------
# parse_dollar_amount
# ---------------------------------------------------------------------------


def test_parse_dollar_amount_strips_dollar_sign():
    series = pl.Series(["$134.09", "$38.48", "$0.00"])
    result = parse_dollar_amount(series)
    assert result.to_list() == [134.09, 38.48, 0.00]
    assert result.dtype == pl.Float64


def test_parse_dollar_amount_single_value():
    result = parse_dollar_amount(pl.Series(["$1.00"]))
    assert result[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# parse_currency_column
# ---------------------------------------------------------------------------


def test_parse_currency_column_strips_dollar_and_comma():
    series = pl.Series(["$24,295", "$1,000,000", "$500"])
    result = parse_currency_column(series)
    assert result.to_list() == [24295.0, 1_000_000.0, 500.0]


# ---------------------------------------------------------------------------
# build_datetime
# ---------------------------------------------------------------------------


def test_build_datetime_returns_datetime_series(sample_transactions):
    result = build_datetime(sample_transactions)
    assert result.dtype == pl.Datetime


def test_build_datetime_correct_values(sample_transactions):
    result = build_datetime(sample_transactions)
    assert result[0] == datetime(2002, 9, 1, 6, 21)
    assert result[1] == datetime(2002, 9, 2, 17, 45)


# ---------------------------------------------------------------------------
# log_bin
# ---------------------------------------------------------------------------


def test_log_bin_returns_correct_number_of_bins():
    series = pl.Series(list(range(1, 101)), dtype=pl.Float64)
    bin_indices, edges = log_bin(series, n_bins=10)
    assert bin_indices.is_between(1, 10).all()
    assert len(edges) == 10


def test_log_bin_indices_are_positive_integers():
    series = pl.Series([10.0, 50.0, 100.0, 200.0])
    bin_indices, _ = log_bin(series, n_bins=4)
    assert (bin_indices >= 1).all()
    assert bin_indices.dtype in (pl.Int32, pl.Int64)


def test_log_bin_edges_length_matches_n_bins():
    series = pl.Series([float(i) for i in range(1, 201)])
    _, edges = log_bin(series, n_bins=20)
    assert len(edges) == 20


# ---------------------------------------------------------------------------
# apply_binning
# ---------------------------------------------------------------------------


def test_apply_binning_adds_bin_columns(sample_transactions_with_datetime):
    result, bin_edges = apply_binning(sample_transactions_with_datetime, bin_specs={"Amount": 5})
    assert "Amount_bin" in result.columns
    assert "Amount" in bin_edges
    assert len(bin_edges["Amount"]) == 5


def test_apply_binning_skips_missing_columns(sample_transactions_with_datetime):
    result, bin_edges = apply_binning(
        sample_transactions_with_datetime, bin_specs={"Amount": 5, "NonExistent": 10}
    )
    assert "NonExistent_bin" not in result.columns
    assert "NonExistent" not in bin_edges


# ---------------------------------------------------------------------------
# build_vocab
# ---------------------------------------------------------------------------


def test_build_vocab_one_indexed():
    series = pl.Series(["a", "b", "c"])
    vocab = build_vocab(series)
    assert set(vocab.values()) == {1, 2, 3}
    assert min(vocab.values()) == 1


def test_build_vocab_handles_duplicates():
    series = pl.Series(["x", "x", "y"])
    vocab = build_vocab(series)
    assert len(vocab) == 2


# ---------------------------------------------------------------------------
# encode_categoricals
# ---------------------------------------------------------------------------


def test_encode_categoricals_adds_id_columns(sample_transactions_with_datetime):
    result, vocabs = encode_categoricals(sample_transactions_with_datetime, ["Use Chip", "MCC"])
    assert "Use Chip_id" in result.columns
    assert "MCC_id" in result.columns
    assert "Use Chip" in vocabs
    assert "MCC" in vocabs


def test_encode_categoricals_ids_are_positive_integers(sample_transactions_with_datetime):
    result, _ = encode_categoricals(sample_transactions_with_datetime, ["Use Chip"])
    assert result["Use Chip_id"].min() >= 1


# ---------------------------------------------------------------------------
# build_behavior_tokens
# ---------------------------------------------------------------------------


def test_build_behavior_tokens_adds_beh_token_column(sample_transactions_with_datetime):
    df, _ = apply_binning(sample_transactions_with_datetime, bin_specs={"Amount": 5})
    df = df.with_columns(pl.col("MCC").cast(pl.Utf8))
    result, token_vocab = build_behavior_tokens(df)
    assert "beh_token" in result.columns
    assert len(token_vocab) > 0


def test_build_behavior_tokens_unique_combos_get_unique_ids(sample_transactions_with_datetime):
    df, _ = apply_binning(sample_transactions_with_datetime, bin_specs={"Amount": 5})
    df = df.with_columns(pl.col("MCC").cast(pl.Utf8))
    _, token_vocab = build_behavior_tokens(df)
    assert len(set(token_vocab.values())) == len(token_vocab)


# ---------------------------------------------------------------------------
# add_temporal_features
# ---------------------------------------------------------------------------


def test_add_temporal_features_adds_column(sample_transactions_with_datetime):
    result = add_temporal_features(sample_transactions_with_datetime)
    assert "hours_since_last_txn" in result.columns


def test_add_temporal_features_first_txn_is_null(sample_transactions_with_datetime):
    result = add_temporal_features(sample_transactions_with_datetime)
    n_groups = result.select(["User", "Card"]).unique().height
    assert result["hours_since_last_txn"].null_count() == n_groups


def test_add_temporal_features_subsequent_txns_non_negative(sample_transactions_with_datetime):
    result = add_temporal_features(sample_transactions_with_datetime)
    non_null = result["hours_since_last_txn"].drop_nulls()
    assert (non_null >= 0).all()


# ---------------------------------------------------------------------------
# assemble_sequences
# ---------------------------------------------------------------------------


@pytest.fixture
def preprocessed_df(sample_transactions_with_datetime):
    df, _ = apply_binning(sample_transactions_with_datetime, bin_specs={"Amount": 5})
    df = df.with_columns(pl.col("MCC").cast(pl.Utf8))
    df, _ = build_behavior_tokens(df)
    return df


def test_assemble_sequences_groups_by_card(preprocessed_df):
    result = assemble_sequences(preprocessed_df, min_seq_length=1)
    assert set(result.columns) >= {"User", "Card", "beh_seq", "seq_length"}
    n_groups = preprocessed_df.select(["User", "Card"]).unique().height
    assert len(result) == n_groups


def test_assemble_sequences_filters_short_sequences(preprocessed_df):
    result = assemble_sequences(preprocessed_df, min_seq_length=MIN_SEQ_LENGTH)
    # All test cards have fewer transactions than MIN_SEQ_LENGTH, so result is empty
    assert len(result) == 0

def test_assemble_sequences_respects_min_seq_length(preprocessed_df):
    result = assemble_sequences(preprocessed_df, min_seq_length=2)
    assert len(result) == 0 or result["seq_length"].min() >= 2


def test_assemble_sequences_beh_seq_is_list(preprocessed_df):
    result = assemble_sequences(preprocessed_df, min_seq_length=1)
    assert all(isinstance(seq, list) for seq in result["beh_seq"].to_list())


# ---------------------------------------------------------------------------
# save_artifacts
# ---------------------------------------------------------------------------


def test_save_artifacts_writes_valid_json(tmp_path):
    artifacts = {
        "bin_edges": {"Amount": [0.0, 50.0, 100.0]},
        "vocabs": {"MCC": {"5300": 1, "5411": 2}},
        "token_vocab": {"1|5300|Swipe Transaction": 1},
    }
    out = tmp_path / "artifacts.json"
    save_artifacts(artifacts, out)
    loaded = json.loads(out.read_text())
    assert loaded["bin_edges"]["Amount"] == [0.0, 50.0, 100.0]
    assert loaded["vocabs"]["MCC"]["5300"] == 1
