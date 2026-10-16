
from pathlib import Path
import argparse
import csv
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

from models import NextWordModel, count_parameters


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed_word"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"

SEED = 42
CONTEXT_LENGTH = 24
BATCH_SIZE = 32
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    train_ids = np.load(DATA_DIR / "train_ids.npy")
    val_ids = np.load(DATA_DIR / "validation_ids.npy")

    with open(DATA_DIR / "vocab.json", encoding="utf-8") as f:
        vocab_data = json.load(f)

    vocab_size = len(vocab_data["word_to_id"])

    return (
        torch.tensor(train_ids, dtype=torch.long),
        torch.tensor(val_ids, dtype=torch.long),
        vocab_size,
    )


def sample_batch(data, batch_size, context_length, device):
    """Sample random contiguous contexts and their next-word targets."""
    if len(data) <= context_length:
        raise ValueError("Dataset is too short for the context length.")

    starts = torch.randint(
        0,
        len(data) - context_length,
        (batch_size,),
    )

    x = torch.stack([
        data[start:start + context_length]
        for start in starts.tolist()
    ])

    y = data[starts + context_length]

    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, data, device, batch_size=256):
    """Calculate exact mean next-word loss over all validation windows."""
    was_training = model.training
    model.eval()

    windows = data.unfold(
        0,
        CONTEXT_LENGTH + 1,
        1,
    )

    total_loss = 0.0
    total_tokens = 0

    for start in range(0, len(windows), batch_size):
        batch = windows[start:start + batch_size]

        x = batch[:, :-1].to(device)
        y = batch[:, -1].to(device)

        logits = model(x)

        loss = nn.functional.cross_entropy(
            logits,
            y,
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += y.numel()

    if was_training:
        model.train()

    return total_loss / total_tokens


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["rnn", "birnn", "lstm", "gru", "all"],
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=100)

    args = parser.parse_args()

    if args.epochs < 1 or args.steps_per_epoch < 1:
        raise ValueError("Epochs and steps per epoch must be positive.")

    set_seed(SEED)

    # CPU execution is intentional for this experiment.
    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    train_data, val_data, vocab_size = load_data()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    architectures = (
        ["rnn", "birnn", "lstm", "gru"]
        if args.model == "all"
        else [args.model]
    )

    summary = []

    for architecture in architectures:
        print(f"\n{'=' * 60}")
        print(f"Training {architecture.upper()}")
        print(f"{'=' * 60}")

        # Reset the seed before each model for reproducibility.
        set_seed(SEED)

        model = NextWordModel(
            vocab_size=vocab_size,
            architecture=architecture,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        criterion = nn.CrossEntropyLoss()
        parameter_count = count_parameters(model)

        print(f"Parameters: {parameter_count:,}")
        print(f"Device: {device}")

        history = []
        best_validation_loss = float("inf")
        start_time = time.perf_counter()

        for epoch in range(1, args.epochs + 1):
            model.train()
            batch_losses = []

            for step in range(1, args.steps_per_epoch + 1):
                x, y = sample_batch(
                    train_data,
                    BATCH_SIZE,
                    CONTEXT_LENGTH,
                    device,
                )

                optimizer.zero_grad(set_to_none=True)

                logits = model(x)
                loss = criterion(logits, y)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                optimizer.step()
                batch_losses.append(loss.item())

            train_loss = float(np.mean(batch_losses))
            val_loss = evaluate(model, val_data, device)

            # Perplexity is the exponential of cross-entropy loss.
            val_perplexity = float(np.exp(min(val_loss, 20)))

            row = {
                "model": architecture,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_perplexity": val_perplexity,
                "parameters": parameter_count,
                "elapsed_seconds": time.perf_counter() - start_time,
            }

            history.append(row)

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f} | "
                f"val perplexity: {val_perplexity:.2f} | "
                f"time: {row['elapsed_seconds']:.1f}s"
            )

            # Build the checkpoint for this epoch.
            checkpoint = {
                "architecture": architecture,
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "embedding_dim": EMBEDDING_DIM,
                "hidden_dim": HIDDEN_DIM,
                "context_length": CONTEXT_LENGTH,
                "epoch": epoch,
                "validation_loss": val_loss,
                "seed": SEED,
            }

            # Always save the latest checkpoint.
            torch.save(
                checkpoint,
                MODELS_DIR / f"{architecture}_latest.pt",
            )

            # Save the best checkpoint only when validation improves.
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss

                torch.save(
                    checkpoint,
                    MODELS_DIR / f"{architecture}_best.pt",
                )

                print(
                    f"  Best checkpoint updated "
                    f"(validation loss: {val_loss:.4f})"
                )

            # Save training history incrementally.
            history_path = RESULTS_DIR / f"{architecture}_history.csv"

            with open(
                history_path,
                "w",
                newline="",
                encoding="utf-8",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(history[0].keys()),
                )
                writer.writeheader()
                writer.writerows(history)

        elapsed = time.perf_counter() - start_time
        best = min(history, key=lambda row: row["validation_loss"])

        summary.append({
            "model": architecture,
            "parameters": parameter_count,
            "epochs_completed": args.epochs,
            "training_seconds": round(elapsed, 2),
            "best_validation_loss": best["validation_loss"],
            "best_validation_perplexity": best["validation_perplexity"],
            "last_validation_loss": history[-1]["validation_loss"],
            "best_epoch": best["epoch"],
        })

        # Save the summary after each model so progress is retained.
        with open(
            RESULTS_DIR / "training_summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2)

        print(f"Finished {architecture.upper()} in {elapsed:.1f}s")
        print(f"Best epoch: {best['epoch']}")
        print(f"Best validation loss: {best['validation_loss']:.4f}")

    print("\nTRAINING SUMMARY")

    for row in summary:
        print(
            f"{row['model'].upper():5s} | "
            f"best epoch: {row['best_epoch']} | "
            f"best val loss: {row['best_validation_loss']:.4f} | "
            f"best val perplexity: "
            f"{row['best_validation_perplexity']:.2f} | "
            f"time: {row['training_seconds']:.1f}s"
        )

    print(f"\nResults saved in: {RESULTS_DIR}")
    print(f"Checkpoints saved in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
