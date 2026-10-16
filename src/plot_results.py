from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["rnn", "birnn", "lstm", "gru"]
MODEL_LABELS = {
    "rnn": "Simple RNN",
    "birnn": "Bidirectional RNN",
    "lstm": "LSTM",
    "gru": "GRU",
}


def load_histories():
    frames = []
    for model in MODEL_ORDER:
        path = RESULTS_DIR / f"{model}_history.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing training history: {path}")
        frame = pd.read_csv(path)
        frame["model"] = model
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def save_line_plot(histories, column, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in MODEL_ORDER:
        data = histories[histories["model"] == model]
        ax.plot(
            data["epoch"],
            data[column],
            marker="o",
            linewidth=2,
            markersize=4,
            label=MODEL_LABELS[model],
        )
    ax.set_xlabel("Training round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    histories = load_histories()

    # Learning curves.
    save_line_plot(
        histories, "train_loss", "Training cross-entropy loss",
        "Training Loss by Model", "training_loss.png"
    )
    save_line_plot(
        histories, "validation_loss", "Validation cross-entropy loss",
        "Validation Loss by Model", "validation_loss.png"
    )
    save_line_plot(
        histories, "validation_perplexity", "Validation perplexity",
        "Validation Perplexity by Model", "validation_perplexity.png"
    )

    summary_path = RESULTS_DIR / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = pd.DataFrame(json.load(f))

    summary["model"] = pd.Categorical(
        summary["model"], categories=MODEL_ORDER, ordered=True
    )
    summary = summary.sort_values("model").copy()
    summary["model_name"] = summary["model"].map(MODEL_LABELS)
    summary.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # Bar plots for the main comparison metrics.
    plot_specs = [
        ("best_validation_loss", "Best validation loss", "best_validation_loss.png"),
        ("best_validation_perplexity", "Best validation perplexity",
         "best_validation_perplexity.png"),
        ("training_seconds", "Training time (seconds)", "training_time.png"),
        ("parameters", "Number of trainable parameters", "parameter_count.png"),
    ]

    for column, ylabel, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = summary["model_name"].astype(str).tolist()
        values = summary[column].astype(float).tolist()
        bars = ax.bar(labels, values)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by Model")
        ax.grid(axis="y", alpha=0.3)
        ax.bar_label(bars, fmt="%.2f" if column != "parameters" else "%.0f",
                     padding=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("Plots saved to:", PLOTS_DIR)
    print("Comparison table saved to:", RESULTS_DIR / "model_comparison.csv")
    print("\nComparison summary:")
    columns = [
        "model_name", "parameters", "best_validation_loss",
        "best_validation_perplexity", "training_seconds"
    ]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
