"""Preprocessing pipeline for the IBM credit card transactions dataset.

Follows the PANTHER approach: log-binning of continuous features, categorical
encoding, composite behavior token construction, and sequence assembly per card.
Reference: https://arxiv.org/html/2510.10102v2
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


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


def load_raw_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the raw CCT CSV files from *data_dir*.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``credit_card_transactions-ibm_v2.csv``,
        ``sd254_cards.csv``, and ``sd254_users.csv``.

    Returns
    -------
    transactions : pd.DataFrame
    cards : pd.DataFrame
    users : pd.DataFrame
    """
    data_dir = Path(data_dir)
    transactions = pd.read_csv(data_dir / "credit_card_transactions-ibm_v2.csv")
    cards = pd.read_csv(data_dir / "sd254_cards.csv")
    users = pd.read_csv(data_dir / "sd254_users.csv")
    return transactions, cards, users


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_dollar_amount(series: pd.Series) -> pd.Series:
    """Strip leading ``$`` and cast to float.

    Parameters
    ----------
    series : pd.Series
        Raw amount strings such as ``"$134.09"``.

    Returns
    -------
    pd.Series
        Float-typed amounts.
    """
    return series.str.lstrip("$").astype(float)


def build_datetime(df: pd.DataFrame) -> pd.Series:
    """Combine Year, Month, Day, and Time columns into a datetime Series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with integer ``Year``, ``Month``, ``Day`` columns and a
        ``Time`` string column (``"HH:MM"`` format).

    Returns
    -------
    pd.Series
        Parsed datetime values.
    """
    dt_str = (
        df["Year"].astype(str)
        + "-"
        + df["Month"].astype(str).str.zfill(2)
        + "-"
        + df["Day"].astype(str).str.zfill(2)
        + " "
        + df["Time"]
    )
    return pd.to_datetime(dt_str, format="%Y-%m-%d %H:%M")


def parse_currency_column(series: pd.Series) -> pd.Series:
    """Strip ``$`` and commas from a currency string column and cast to float.

    Parameters
    ----------
    series : pd.Series
        Currency strings such as ``"$24,295"`` or ``"$24295"``.

    Returns
    -------
    pd.Series
        Float-typed values.
    """
    return series.str.replace(r"[$,]", "", regex=True).astype(float)


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def log_bin(series: pd.Series, n_bins: int) -> tuple[pd.Series, list[float]]:
    """Equal-frequency log-space binning.

    Splits *series* into *n_bins* equal-frequency buckets after a log
    transform, then assigns 1-indexed bin labels back in the original space.

    Parameters
    ----------
    series : pd.Series
        Positive numeric values to bin.
    n_bins : int
        Number of equal-frequency buckets.

    Returns
    -------
    bin_indices : pd.Series
        Integer bin labels (1 … n_bins).
    bin_edges : list of float
        Left edge of each bucket in original scale (length = n_bins).
    """
    log_vals = np.log1p(series.clip(lower=0))
    sorted_log = np.sort(log_vals.dropna().values)
    buckets = np.array_split(sorted_log, n_bins)
    edges = [float(np.expm1(b[0])) for b in buckets if len(b) > 0]
    bin_indices = pd.cut(
        series,
        bins=[-np.inf] + edges[1:] + [np.inf],
        labels=False,
        right=False,
        duplicates="drop",
    ).fillna(0).astype(int) + 1
    return bin_indices, edges


def apply_binning(
    df: pd.DataFrame,
    bin_specs: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Apply :func:`log_bin` to all configured numeric columns in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that contains the columns listed in *bin_specs*.
    bin_specs : dict, optional
        Mapping of ``column_name -> n_bins``. Defaults to
        :data:`NUM_BIN_COLUMNS`.

    Returns
    -------
    df : pd.DataFrame
        Copy of *df* with binned columns suffixed ``_bin``.
    bin_edges : dict
        Mapping of column name to its bin edge list.
    """
    if bin_specs is None:
        bin_specs = NUM_BIN_COLUMNS
    df = df.copy()
    bin_edges: dict[str, list[float]] = {}
    for col, n_bins in bin_specs.items():
        if col in df.columns:
            df[f"{col}_bin"], bin_edges[col] = log_bin(df[col], n_bins)
    return df, bin_edges


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------


def build_vocab(series: pd.Series) -> dict[str, int]:
    """Assign 1-indexed integer IDs to every unique value in *series*.

    Parameters
    ----------
    series : pd.Series
        Categorical values.

    Returns
    -------
    dict
        Mapping of string value to integer ID (1-indexed; 0 reserved for
        unknown).
    """
    unique_vals = sorted(series.dropna().astype(str).unique())
    return {v: idx + 1 for idx, v in enumerate(unique_vals)}


def encode_categoricals(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Encode categorical columns to integer IDs in-place (new ``_id`` cols).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Column names to encode.

    Returns
    -------
    df : pd.DataFrame
        Copy of *df* with ``<col>_id`` integer columns added.
    vocabs : dict
        Mapping of column name to its ``{value: id}`` vocabulary dict.
    """
    df = df.copy()
    vocabs: dict[str, dict[str, int]] = {}
    for col in columns:
        if col in df.columns:
            vocab = build_vocab(df[col])
            vocabs[col] = vocab
            df[f"{col}_id"] = df[col].astype(str).map(vocab).fillna(0).astype(int)
    return df, vocabs


# ---------------------------------------------------------------------------
# Composite behavior tokens
# ---------------------------------------------------------------------------


def build_behavior_tokens(
    df: pd.DataFrame,
    seq_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Create composite token IDs from the Cartesian product of *seq_columns*.

    Follows the PANTHER tokenization strategy: each unique combination of
    (Amount_bin, MCC, Use Chip) receives a single integer token ID.  The
    ``_bin`` suffixed version of ``Amount`` is used automatically.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that already contains ``Amount_bin``, ``MCC``, and
        ``Use Chip`` columns (or whatever *seq_columns* specifies).
    seq_columns : list of str, optional
        Behavior feature columns to combine. Defaults to :data:`SEQ_COLUMNS`.

    Returns
    -------
    df : pd.DataFrame
        Copy of *df* with a ``beh_token`` integer column.
    token_vocab : dict
        Mapping of ``"Amount_bin|MCC|Use Chip"`` string to token ID.
    """
    if seq_columns is None:
        seq_columns = SEQ_COLUMNS

    resolved: list[str] = []
    for col in seq_columns:
        resolved.append(f"{col}_bin" if col == "Amount" else col)

    combo = df[resolved].astype(str).agg("|".join, axis=1)
    unique_combos = sorted(combo.dropna().unique())
    token_vocab = {v: idx + 1 for idx, v in enumerate(unique_combos)}
    df = df.copy()
    df["beh_token"] = combo.map(token_vocab).fillna(0).astype(int)
    return df, token_vocab


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hours-since-last-transaction per card to *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``datetime``, ``User``, and ``Card`` columns, sorted
        chronologically within each card.

    Returns
    -------
    pd.DataFrame
        Copy with a ``hours_since_last_txn`` float column (NaN for the first
        transaction of each card).
    """
    df = df.copy()
    df = df.sort_values(["User", "Card", "datetime"])
    delta = df.groupby(["User", "Card"])["datetime"].diff()
    df["hours_since_last_txn"] = delta.dt.total_seconds() / 3600
    return df


# ---------------------------------------------------------------------------
# Merge and sequence assembly
# ---------------------------------------------------------------------------


def merge_datasets(
    transactions: pd.DataFrame,
    cards: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.DataFrame:
    """Join transactions with card and user metadata.

    Parameters
    ----------
    transactions : pd.DataFrame
        Raw transactions (from :func:`load_raw_data`).
    cards : pd.DataFrame
        Card metadata keyed on ``User`` and ``CARD INDEX``.
    users : pd.DataFrame
        User metadata keyed on row position matching ``User`` ID.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    users = users.copy()
    users.index.name = "User"
    users = users.reset_index()

    merged = transactions.merge(
        cards,
        left_on=["User", "Card"],
        right_on=["User", "CARD INDEX"],
        how="left",
    )
    merged = merged.merge(users, on="User", how="left")
    return merged


def assemble_sequences(
    df: pd.DataFrame,
    min_seq_length: int = MIN_SEQ_LENGTH,
) -> pd.DataFrame:
    """Group transactions into per-card sequences and filter short histories.

    Each row in the output represents one card's full transaction history.
    In addition to the composite behavior token sequence, individual encoded
    feature sequences (``Amount_bin``, ``MCC_id``, ``Use Chip_id``) and
    per-card attribute scalars are included when present.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed transaction-level DataFrame with ``User``, ``Card``,
        ``datetime``, ``beh_token``, and ``Is Fraud?`` columns.
    min_seq_length : int
        Cards with fewer transactions than this threshold are dropped.

    Returns
    -------
    pd.DataFrame
        One row per card with list-valued sequence columns and scalar attrs.
    """
    df = df.sort_values(["User", "Card", "datetime"])

    seq_aggs: dict = {
        "beh_seq": pd.NamedAgg(column="beh_token", aggfunc=list),
        "timestamps": pd.NamedAgg(column="datetime", aggfunc=list),
        "fraud_labels": pd.NamedAgg(column="Is Fraud?", aggfunc=list),
        "seq_length": pd.NamedAgg(column="beh_token", aggfunc="count"),
    }
    for col in BEH_FEATURE_SEQ_COLUMNS:
        if col in df.columns:
            seq_aggs[f"{col}_seq"] = pd.NamedAgg(column=col, aggfunc=list)

    attr_aggs: dict = {}
    for col in USER_ATTR_ID_COLUMNS:
        if col in df.columns:
            attr_aggs[col] = pd.NamedAgg(column=col, aggfunc="first")

    grouped = (
        df.groupby(["User", "Card"])
        .agg(**{**seq_aggs, **attr_aggs})
        .reset_index()
    )
    return grouped[grouped["seq_length"] >= min_seq_length].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def preprocess(
    data_dir: str | Path,
    min_seq_length: int = MIN_SEQ_LENGTH,
    pretrain: bool = False,
) -> tuple[pd.DataFrame, dict]:
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
    sequences : pd.DataFrame
        Per-card sequence DataFrame ready for model input.
    artifacts : dict
        Encoding artifacts (bin edges, vocabularies, token vocab) needed to
        encode new transactions at inference time.
    """
    transactions, cards, users = load_raw_data(data_dir)

    # --- parse raw formats ---
    transactions["Amount"] = parse_dollar_amount(transactions["Amount"])
    transactions["datetime"] = build_datetime(transactions)
    cards["Credit Limit"] = parse_currency_column(cards["Credit Limit"])
    users["Per Capita Income - Zipcode"] = parse_currency_column(
        users["Per Capita Income - Zipcode"]
    )
    users["Yearly Income - Person"] = parse_currency_column(
        users["Yearly Income - Person"]
    )

    # --- merge ---
    df = merge_datasets(transactions, cards, users)

    # --- optional pretrain filter ---
    if pretrain:
        df = df[df["Is Fraud?"] == "No"].copy()

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
