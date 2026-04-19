"""Preprocessing pipeline for the IBM credit card transactions dataset.

Follows the PANTHER approach: log-binning of continuous features, categorical
encoding, composite behavior token construction, and sequence assembly per card.
Reference: https://arxiv.org/html/2510.10102v2
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEQ_COLUMNS: list[str] = ["Amount", "MCC", "Use Chip"]
ELEMENT_COLUMNS: list[str] = [
    "Card Brand",
    "Card Type",
    "Has Chip",
    "Cards Issued",
    "Card on Dark Web",
    "Gender",
    "FICO Score",
    "Num Credit Cards",
]
NUM_BIN_COLUMNS: dict[str, int] = {
    "Amount": 100,
    "Credit Limit": 100,
    "Per Capita Income - Zipcode": 100,
    "Yearly Income - Person": 100,
}
MIN_SEQ_LENGTH: int = 30

# Per-transaction feature columns stored as sequences (produced after binning/encoding)
BEH_FEATURE_SEQ_COLUMNS: list[str] = ["Amount_bin", "MCC_id", "Use Chip_id"]
# Per-card scalar attribute columns (encoded; produced after encode_categoricals)
USER_ATTR_ID_COLUMNS: list[str] = [f"{c}_id" for c in ELEMENT_COLUMNS]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw_data(data_dir: str | Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load the raw CCT CSV files from *data_dir*.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``credit_card_transactions-ibm_v2.csv``,
        ``sd254_cards.csv``, and ``sd254_users.csv``.

    Returns
    -------
    transactions : pl.DataFrame
    cards : pl.DataFrame
    users : pl.DataFrame
    """
    data_dir = Path(data_dir)
    transactions = pl.read_csv(data_dir / "credit_card_transactions-ibm_v2.csv", infer_schema_length=10000)
    cards = pl.read_csv(data_dir / "sd254_cards.csv", infer_schema_length=10000)
    users = pl.read_csv(data_dir / "sd254_users.csv", infer_schema_length=10000)
    return transactions, cards, users


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_dollar_amount(series: pl.Series) -> pl.Series:
    """Strip leading ``$`` and cast to float.

    Parameters
    ----------
    series : pl.Series
        Raw amount strings such as ``"$134.09"``.

    Returns
    -------
    pl.Series
        Float-typed amounts.
    """
    return series.str.strip_chars_start("$").cast(pl.Float64)


def build_datetime(df: pl.DataFrame) -> pl.Series:
    """Combine Year, Month, Day, and Time columns into a datetime Series.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with integer ``Year``, ``Month``, ``Day`` columns and a
        ``Time`` string column (``"HH:MM"`` format).

    Returns
    -------
    pl.Series
        Parsed datetime values.
    """
    dt_str = (
        df["Year"].cast(pl.Utf8)
        + "-"
        + df["Month"].cast(pl.Utf8).str.zfill(2)
        + "-"
        + df["Day"].cast(pl.Utf8).str.zfill(2)
        + " "
        + df["Time"]
    )
    return dt_str.str.to_datetime(format="%Y-%m-%d %H:%M", strict=False)


def parse_currency_column(series: pl.Series) -> pl.Series:
    """Strip ``$`` and commas from a currency string column and cast to float.

    Parameters
    ----------
    series : pl.Series
        Currency strings such as ``"$24,295"`` or ``"$24295"``.

    Returns
    -------
    pl.Series
        Float-typed values.
    """
    return series.str.replace_all(r"[$,]", "").cast(pl.Float64)


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def log_bin(series: pl.Series, n_bins: int) -> tuple[pl.Series, list[float]]:
    """Equal-frequency log-space binning.

    Splits *series* into *n_bins* equal-frequency buckets after a log
    transform, then assigns 1-indexed bin labels back in the original space.

    Parameters
    ----------
    series : pl.Series
        Positive numeric values to bin.
    n_bins : int
        Number of equal-frequency buckets.

    Returns
    -------
    bin_indices : pl.Series
        Integer bin labels (1 … n_bins).
    bin_edges : list of float
        Left edge of each bucket in original scale (length = n_bins).
    """
    arr = series.to_numpy(allow_copy=True).astype(float)
    log_vals = np.log1p(np.clip(arr, 0, None))
    valid = np.sort(log_vals[~np.isnan(log_vals)])
    buckets = np.array_split(valid, n_bins)
    edges = [float(np.expm1(b[0])) for b in buckets if len(b) > 0]
    # np.digitize gives 0-indexed position among boundaries edges[1:]
    # adding 1 converts to 1-indexed bins; nan positions stay 0
    bin_indices = np.digitize(arr, edges[1:], right=False) + 1
    bin_indices = np.where(np.isnan(arr), 0, bin_indices).astype(np.int32)
    return pl.Series(values=bin_indices), edges


def apply_binning(
    df: pl.DataFrame,
    bin_specs: dict[str, int] | None = None,
) -> tuple[pl.DataFrame, dict[str, list[float]]]:
    """Apply :func:`log_bin` to all configured numeric columns in *df*.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame that contains the columns listed in *bin_specs*.
    bin_specs : dict, optional
        Mapping of ``column_name -> n_bins``. Defaults to
        :data:`NUM_BIN_COLUMNS`.

    Returns
    -------
    df : pl.DataFrame
        DataFrame with binned columns suffixed ``_bin``.
    bin_edges : dict
        Mapping of column name to its bin edge list.
    """
    if bin_specs is None:
        bin_specs = NUM_BIN_COLUMNS
    bin_edges: dict[str, list[float]] = {}
    for col, n_bins in bin_specs.items():
        if col in df.columns:
            bin_series, edges = log_bin(df[col], n_bins)
            df = df.with_columns(bin_series.alias(f"{col}_bin"))
            bin_edges[col] = edges
    return df, bin_edges


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------


def build_vocab(series: pl.Series) -> dict[str, int]:
    """Assign 1-indexed integer IDs to every unique value in *series*.

    Parameters
    ----------
    series : pl.Series
        Categorical values.

    Returns
    -------
    dict
        Mapping of string value to integer ID (1-indexed; 0 reserved for
        unknown).
    """
    unique_vals = sorted(series.drop_nulls().cast(pl.Utf8).unique().to_list())
    return {v: idx + 1 for idx, v in enumerate(unique_vals)}


def encode_categoricals(
    df: pl.DataFrame,
    columns: list[str],
) -> tuple[pl.DataFrame, dict[str, dict[str, int]]]:
    """Encode categorical columns to integer IDs (new ``_id`` cols).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    columns : list of str
        Column names to encode.

    Returns
    -------
    df : pl.DataFrame
        DataFrame with ``<col>_id`` integer columns added.
    vocabs : dict
        Mapping of column name to its ``{value: id}`` vocabulary dict.
    """
    vocabs: dict[str, dict[str, int]] = {}
    for col in columns:
        if col in df.columns:
            vocab = build_vocab(df[col])
            vocabs[col] = vocab
            old_vals = pl.Series(list(vocab.keys()), dtype=pl.Utf8)
            new_vals = pl.Series(list(vocab.values()), dtype=pl.Int64)
            df = df.with_columns(
                df[col].cast(pl.Utf8)
                .replace_strict(old_vals, new_vals, default=0)
                .cast(pl.Int64)
                .alias(f"{col}_id")
            )
    return df, vocabs


# ---------------------------------------------------------------------------
# Composite behavior tokens
# ---------------------------------------------------------------------------


def build_behavior_tokens(
    df: pl.DataFrame,
    seq_columns: list[str] | None = None,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Create composite token IDs from the Cartesian product of *seq_columns*.

    Follows the PANTHER tokenization strategy: each unique combination of
    (Amount_bin, MCC, Use Chip) receives a single integer token ID.  The
    ``_bin`` suffixed version of ``Amount`` is used automatically.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame that already contains ``Amount_bin``, ``MCC``, and
        ``Use Chip`` columns (or whatever *seq_columns* specifies).
    seq_columns : list of str, optional
        Behavior feature columns to combine. Defaults to :data:`SEQ_COLUMNS`.

    Returns
    -------
    df : pl.DataFrame
        DataFrame with a ``beh_token`` integer column.
    token_vocab : dict
        Mapping of ``"Amount_bin|MCC|Use Chip"`` string to token ID.
    """
    if seq_columns is None:
        seq_columns = SEQ_COLUMNS

    resolved = [f"{col}_bin" if col == "Amount" else col for col in seq_columns]

    combo_expr = pl.concat_str([pl.col(c).cast(pl.Utf8) for c in resolved], separator="|")
    df = df.with_columns(combo_expr.alias("_combo"))

    unique_combos = sorted(df["_combo"].drop_nulls().unique().to_list())
    token_vocab = {v: idx + 1 for idx, v in enumerate(unique_combos)}

    old_vals = pl.Series(list(token_vocab.keys()), dtype=pl.Utf8)
    new_vals = pl.Series(list(token_vocab.values()), dtype=pl.Int64)
    df = df.with_columns(
        pl.col("_combo").replace_strict(old_vals, new_vals, default=0).cast(pl.Int64).alias("beh_token")
    ).drop("_combo")

    return df, token_vocab


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------


def add_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add hours-since-last-transaction per card to *df*.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain ``datetime``, ``User``, and ``Card`` columns.

    Returns
    -------
    pl.DataFrame
        Sorted by ``(User, Card, datetime)`` with a ``hours_since_last_txn``
        float column (null for the first transaction of each card).
    """
    return df.sort(["User", "Card", "datetime"]).with_columns(
        (
            pl.col("datetime").diff().over(["User", "Card"])
            .dt.total_seconds()
            .cast(pl.Float64)
            / 3600
        ).alias("hours_since_last_txn")
    )


# ---------------------------------------------------------------------------
# Merge and sequence assembly
# ---------------------------------------------------------------------------


def merge_datasets(
    transactions: pl.DataFrame,
    cards: pl.DataFrame,
    users: pl.DataFrame,
) -> pl.DataFrame:
    """Join transactions with card and user metadata.

    Parameters
    ----------
    transactions : pl.DataFrame
        Raw transactions (from :func:`load_raw_data`).
    cards : pl.DataFrame
        Card metadata keyed on ``User`` and ``CARD INDEX``.
    users : pl.DataFrame
        User metadata keyed on row position matching ``User`` ID.

    Returns
    -------
    pl.DataFrame
        Merged DataFrame.
    """
    users = users.with_row_index("User").with_columns(pl.col("User").cast(pl.Int64))
    transactions = transactions.with_columns(pl.col("User").cast(pl.Int64))

    merged = transactions.join(
        cards,
        left_on=["User", "Card"],
        right_on=["User", "CARD INDEX"],
        how="left",
    )
    return merged.join(users, on="User", how="left")


def assemble_sequences(
    df: pl.DataFrame,
    min_seq_length: int = MIN_SEQ_LENGTH,
) -> pl.DataFrame:
    """Group transactions into per-card sequences and filter short histories.

    Each row in the output represents one card's full transaction history.
    In addition to the composite behavior token sequence, individual encoded
    feature sequences (``Amount_bin``, ``MCC_id``, ``Use Chip_id``) and
    per-card attribute scalars are included when present.

    Parameters
    ----------
    df : pl.DataFrame
        Preprocessed transaction-level DataFrame with ``User``, ``Card``,
        ``datetime``, ``beh_token``, and ``Is Fraud?`` columns.
    min_seq_length : int
        Cards with fewer transactions than this threshold are dropped.

    Returns
    -------
    pl.DataFrame
        One row per card with list-valued sequence columns and scalar attrs.
    """
    df = df.sort(["User", "Card", "datetime"])

    agg_exprs: list = [
        pl.col("beh_token").alias("beh_seq"),
        pl.col("datetime").alias("timestamps"),
        pl.col("Is Fraud?").alias("fraud_labels"),
        pl.col("beh_token").len().alias("seq_length"),
    ]
    for col in BEH_FEATURE_SEQ_COLUMNS:
        if col in df.columns:
            agg_exprs.append(pl.col(col).alias(f"{col}_seq"))
    for col in USER_ATTR_ID_COLUMNS:
        if col in df.columns:
            agg_exprs.append(pl.col(col).first())

    return (
        df.group_by(["User", "Card"], maintain_order=True)
        .agg(agg_exprs)
        .filter(pl.col("seq_length") >= min_seq_length)
    )


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def preprocess(
    data_dir: str | Path,
    min_seq_length: int = MIN_SEQ_LENGTH,
    pretrain: bool = False,
) -> tuple[pl.DataFrame, dict]:
    """End-to-end preprocessing pipeline for the CCT dataset.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the three raw CSV files.
    min_seq_length : int
        Minimum transactions per card to keep in the sequence dataset.
    pretrain : bool
        When ``True``, drop all fraudulent transactions (for self-supervised
        pretraining, consistent with the PANTHER paper).

    Returns
    -------
    sequences : pl.DataFrame
        Per-card sequence DataFrame ready for model input.
    artifacts : dict
        Encoding artifacts (bin edges, vocabularies, token vocab) needed to
        encode new transactions at inference time.
    """
    transactions, cards, users = load_raw_data(data_dir)

    # --- parse raw formats ---
    transactions = transactions.with_columns(
        parse_dollar_amount(transactions["Amount"]).alias("Amount"),
        build_datetime(transactions).alias("datetime"),
    )
    cards = cards.with_columns(
        parse_currency_column(cards["Credit Limit"]).alias("Credit Limit"),
    )
    users = users.with_columns(
        parse_currency_column(users["Per Capita Income - Zipcode"]).alias("Per Capita Income - Zipcode"),
        parse_currency_column(users["Yearly Income - Person"]).alias("Yearly Income - Person"),
    )

    # --- merge ---
    df = merge_datasets(transactions, cards, users)

    # --- optional pretrain filter ---
    if pretrain:
        df = df.filter(pl.col("Is Fraud?") == "No")

    # --- numeric binning ---
    df, bin_edges = apply_binning(df)

    # --- categorical encoding ---
    cat_cols = ELEMENT_COLUMNS + ["MCC", "Use Chip"]
    df, vocabs = encode_categoricals(df, cat_cols)

    # --- composite behavior tokens ---
    df, token_vocab = build_behavior_tokens(df)

    # --- temporal features ---
    df = add_temporal_features(df)

    # --- sequence assembly ---
    sequences = assemble_sequences(df, min_seq_length=min_seq_length)

    artifacts = {
        "bin_edges": bin_edges,
        "vocabs": vocabs,
        "token_vocab": token_vocab,
        "beh_feature_seq_columns": BEH_FEATURE_SEQ_COLUMNS,
        "user_attr_id_columns": [c for c in USER_ATTR_ID_COLUMNS if c in df.columns],
    }
    return sequences, artifacts


def save_artifacts(artifacts: dict, path: str | Path) -> None:
    """Persist encoding artifacts to a JSON file.

    Parameters
    ----------
    artifacts : dict
        Output of :func:`preprocess` ``artifacts`` dict.
    path : str or Path
        Destination file path (JSON).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "bin_edges": artifacts["bin_edges"],
        "vocabs": artifacts["vocabs"],
        "token_vocab": artifacts["token_vocab"],
    }
    path.write_text(json.dumps(serializable, indent=2))
