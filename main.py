"""End-to-end pipeline: preprocessing → pretraining → fraud fine-tuning → evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import FraudDataset, TransactionDataset
from src.model import FraudHead, build_model_from_artifacts
from src.preprocessing import preprocess, save_artifacts
from src.train import evaluate_fraud, evaluate_retrieval, finetune_fraud, pretrain


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PANTHER fraud detection pipeline")
    p.add_argument("--data-dir", default="data/", help="Directory containing raw CSVs")
    p.add_argument("--checkpoint-dir", default="checkpoints/", help="Model checkpoint directory")
    p.add_argument("--min-seq-length", type=int, default=30)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--fraud-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--freeze-encoder", action="store_true", default=True)
    p.add_argument("--skip-pretrain", action="store_true", help="Load existing checkpoint instead")
    return p.parse_args()


def _make_loader(dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    print("\n=== Preprocessing (pretrain) ===")
    pretrain_sequences, artifacts = preprocess(
        args.data_dir, min_seq_length=args.min_seq_length, pretrain=True
    )
    save_artifacts(artifacts, checkpoint_dir / "artifacts.json")
    print(f"Pretrain sequences: {len(pretrain_sequences):,} cards")

    print("\n=== Preprocessing (fraud) ===")
    fraud_sequences, _ = preprocess(
        args.data_dir, min_seq_length=args.min_seq_length, pretrain=False
    )
    print(f"Fraud sequences: {len(fraud_sequences):,} cards")

    # ------------------------------------------------------------------
    # Splits
    # Card-level splits ensure no card appears in more than one partition.
    # ------------------------------------------------------------------

    # Pretrain: 80 / 10 / 10
    pt_train, pt_tmp = train_test_split(pretrain_sequences, test_size=0.2, random_state=42)
    pt_val, pt_test = train_test_split(pt_tmp, test_size=0.5, random_state=42)
    print(
        f"\nPretrain split  — train: {len(pt_train):,}  val: {len(pt_val):,}  test: {len(pt_test):,}"
    )

    # Fraud: 70 / 15 / 15, stratified by whether a card has any fraud transaction
    fraud_strata = fraud_sequences["fraud_labels"].apply(lambda x: 1 if "Yes" in x else 0)
    fd_train, fd_tmp = train_test_split(
        fraud_sequences, test_size=0.3, random_state=42, stratify=fraud_strata
    )
    fd_tmp_strata = fd_tmp["fraud_labels"].apply(lambda x: 1 if "Yes" in x else 0)
    fd_val, fd_test = train_test_split(
        fd_tmp, test_size=0.5, random_state=42, stratify=fd_tmp_strata
    )
    print(
        f"Fraud split     — train: {len(fd_train):,}  val: {len(fd_val):,}  test: {len(fd_test):,}"
    )

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = build_model_from_artifacts(artifacts, max_seq_len=args.max_seq_length)
    fraud_head = FraudHead(input_dim=model.emb_dim + model._item_emb_dim)

    # ------------------------------------------------------------------
    # Pretraining
    # ------------------------------------------------------------------
    pretrain_checkpoint = checkpoint_dir / "pretrain" / "panther_best.pt"

    if args.skip_pretrain and pretrain_checkpoint.exists():
        print(f"\n=== Loading pretrained checkpoint from {pretrain_checkpoint} ===")
        model.load_state_dict(torch.load(pretrain_checkpoint, map_location=device))
    else:
        print("\n=== Pretraining ===")
        pt_train_loader = _make_loader(
            TransactionDataset(pt_train.reset_index(drop=True), max_length=args.max_seq_length),
            args.batch_size, shuffle=True,
        )
        pt_val_loader = _make_loader(
            TransactionDataset(pt_val.reset_index(drop=True), max_length=args.max_seq_length),
            args.batch_size,
        )
        pretrain(
            model,
            pt_train_loader,
            pt_val_loader,
            num_epochs=args.pretrain_epochs,
            learning_rate=args.learning_rate,
            checkpoint_dir=checkpoint_dir / "pretrain",
            device=device,
        )

    # ------------------------------------------------------------------
    # Retrieval evaluation on held-out pretrain test set
    # ------------------------------------------------------------------
    print("\n=== Retrieval Evaluation (test set) ===")
    pt_test_loader = _make_loader(
        TransactionDataset(pt_test.reset_index(drop=True), max_length=args.max_seq_length),
        args.batch_size,
    )
    retrieval_metrics = evaluate_retrieval(model, pt_test_loader, top_k=(1, 10, 100), device=device)
    for k, v in retrieval_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ------------------------------------------------------------------
    # Fraud fine-tuning (train + val only; test set never seen during training)
    # ------------------------------------------------------------------
    print("\n=== Fraud Fine-tuning ===")
    fd_train_loader = _make_loader(
        FraudDataset(fd_train.reset_index(drop=True), max_history_length=args.max_seq_length),
        args.batch_size, shuffle=True,
    )
    fd_val_loader = _make_loader(
        FraudDataset(fd_val.reset_index(drop=True), max_history_length=args.max_seq_length),
        args.batch_size,
    )
    finetune_fraud(
        model,
        fraud_head,
        fd_train_loader,
        fd_val_loader,
        num_epochs=args.fraud_epochs,
        learning_rate=args.learning_rate,
        freeze_encoder=args.freeze_encoder,
        checkpoint_dir=checkpoint_dir / "fraud",
        device=device,
    )

    # ------------------------------------------------------------------
    # Final fraud evaluation on held-out test set
    # ------------------------------------------------------------------
    print("\n=== Final Fraud Evaluation (test set) ===")
    fd_test_loader = _make_loader(
        FraudDataset(fd_test.reset_index(drop=True), max_history_length=args.max_seq_length),
        args.batch_size,
    )
    fraud_metrics = evaluate_fraud(model, fraud_head, fd_test_loader, device=device)
    for k, v in fraud_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
