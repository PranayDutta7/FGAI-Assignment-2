
from pathlib import Path
import csv
import json

import numpy as np
import torch

from models import NextWordModel


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed_word"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

ARCHITECTURES = ["rnn", "birnn", "lstm", "gru"]
BATCH_SIZE = 256


def main():
    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    # Load the same held-out validation tokens used during training.
    validation_ids = np.load(DATA_DIR / "validation_ids.npy")

    with open(DATA_DIR / "vocab.json", encoding="utf-8") as f:
        vocab_data = json.load(f)

    vocab_size = len(vocab_data["word_to_id"])
    validation_data = torch.tensor(
        validation_ids,
        dtype=torch.long,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for architecture in ARCHITECTURES:
        checkpoint_path = MODELS_DIR / f"{architecture}_best.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint_path}\n"
                "Make sure all four models have been trained."
            )

        # Load the best checkpoint for this architecture.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        context_length = checkpoint["context_length"]

        model = NextWordModel(
            vocab_size=checkpoint["vocab_size"],
            architecture=checkpoint["architecture"],
            embedding_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"],
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Each row contains the context plus its correct next word.
        windows = validation_data.unfold(
            0,
            context_length + 1,
            1,
        )

        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for start in range(0, len(windows), BATCH_SIZE):
                batch = windows[start:start + BATCH_SIZE]

                inputs = batch[:, :-1].to(device)
                targets = batch[:, -1].to(device)

                logits = model(inputs)

                # Choose the single most likely next word.
                predictions = torch.argmax(logits, dim=1)

                correct_predictions += (
                    predictions == targets
                ).sum().item()

                total_predictions += targets.numel()

        accuracy = (
            100.0 * correct_predictions / total_predictions
        )

        row = {
            "model": architecture,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "validation_accuracy_percent": round(accuracy, 4),
            "best_epoch": checkpoint["epoch"],
            "checkpoint_validation_loss": checkpoint["validation_loss"],
        }

        results.append(row)

        print(
            f"{architecture.upper():5s} | "
            f"Correct: {correct_predictions:,}/{total_predictions:,} | "
            f"Accuracy: {accuracy:.4f}%"
        )

    # Save detailed accuracy results.
    accuracy_path = RESULTS_DIR / "next_word_accuracy.csv"

    with open(
        accuracy_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    # Add accuracy to the existing comparison table, if present.
    comparison_path = RESULTS_DIR / "model_comparison.csv"

    if comparison_path.exists():
        with open(
            comparison_path,
            newline="",
            encoding="utf-8",
        ) as f:
            reader = csv.DictReader(f)
            comparison_rows = list(reader)
            fieldnames = list(reader.fieldnames or [])

        if "validation_accuracy_percent" not in fieldnames:
            fieldnames.append("validation_accuracy_percent")

        accuracy_lookup = {
            row["model"]: row["validation_accuracy_percent"]
            for row in results
        }

        for row in comparison_rows:
            model_name = row.get("model_name", "")
            model_key = {
                "Simple RNN": "rnn",
                "Bidirectional RNN": "birnn",
                "LSTM": "lstm",
                "GRU": "gru",
            }.get(model_name)

            if model_key:
                row["validation_accuracy_percent"] = (
                    accuracy_lookup[model_key]
                )

        with open(
            comparison_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)

        print(f"Updated comparison table: {comparison_path}")

    print(f"\nDetailed accuracy results saved to: {accuracy_path}")


if __name__ == "__main__":
    main()
