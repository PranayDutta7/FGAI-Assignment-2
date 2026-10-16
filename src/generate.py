
import argparse
import json
import re
from pathlib import Path

import torch

from models import NextWordModel


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed_word"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "results" / "generated_text"

SEED = 42


def tokenize(text):
    """Use the same word-level tokenization as preprocessing."""
    return re.findall(
        r"\w+(?:['’]\w+)*|[^\w\s]",
        text,
        flags=re.UNICODE,
    )


def load_vocabulary():
    """Load the vocabulary while supporting common JSON layouts."""
    with open(DATA_DIR / "vocab.json", "r", encoding="utf-8") as f:
        saved = json.load(f)

    # Support either token -> ID or ID -> token representations.
    if isinstance(saved, dict) and "token_to_id" in saved:
        token_to_id = saved["token_to_id"]
    elif isinstance(saved, dict) and "word_to_id" in saved:
        token_to_id = saved["word_to_id"]
    elif isinstance(saved, dict) and "stoi" in saved:
        token_to_id = saved["stoi"]
    elif isinstance(saved, dict) and all(
        isinstance(v, int) for v in saved.values()
    ):
        token_to_id = saved
    elif isinstance(saved, dict) and "id_to_token" in saved:
        token_to_id = {
            token: int(index)
            for index, token in enumerate(saved["id_to_token"])
        }
    elif isinstance(saved, dict) and "itos" in saved:
        token_to_id = {
            token: int(index)
            for index, token in enumerate(saved["itos"])
        }
    else:
        raise ValueError(
            "Unrecognized vocabulary format in data/processed_word/vocab.json"
        )

    token_to_id = {
        str(token): int(index)
        for token, index in token_to_id.items()
    }

    id_to_token = {
        index: token for token, index in token_to_id.items()
    }

    if "<UNK>" not in token_to_id:
        raise ValueError("Vocabulary does not contain the required <UNK> token.")

    return token_to_id, id_to_token


def detokenize(tokens):
    """Join word tokens while restoring common punctuation spacing."""
    text = " ".join(tokens)

    # Remove spaces before punctuation.
    text = re.sub(r"\s+([.,!?;:%…])", r"\1", text)

    # Remove spaces around apostrophes when they occur separately.
    text = re.sub(r"\s+(['’])\s*", r"\1", text)
    text = re.sub(r"\s+(['’])", r"\1", text)

    # Restore brackets and parentheses.
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\[\s+", "[", text)
    text = re.sub(r"\{\s+", "{", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)

    # Remove spaces after opening quotation marks.
    text = re.sub(r"(['“])\s+", r"\1", text)

    return text.strip()


def generate(
    model,
    prompt,
    token_to_id,
    id_to_token,
    context_length,
    num_tokens,
    temperature,
    device,
    seed,
):
    """Generate one word at a time using autoregressive sampling."""
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    prompt_tokens = tokenize(prompt)
    if not prompt_tokens:
        raise ValueError("The prompt must contain at least one token.")

    unk_id = token_to_id["<UNK>"]
    generated_ids = [
        token_to_id.get(token, unk_id)
        for token in prompt_tokens
    ]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    model.eval()

    with torch.no_grad():
        for _ in range(num_tokens):
            context_ids = generated_ids[-context_length:]
            x = torch.tensor(
                [context_ids],
                dtype=torch.long,
                device=device,
            )

            logits = model(x)[0]
            probabilities = torch.softmax(
                logits / temperature,
                dim=-1,
            )

            # Sample on CPU for reproducible sampling.
            next_id = torch.multinomial(
                probabilities.cpu(),
                num_samples=1,
                generator=generator,
            ).item()

            generated_ids.append(next_id)

    generated_tokens = [
        id_to_token.get(index, "<UNK>")
        for index in generated_ids
    ]

    return detokenize(generated_tokens)


def main():
    parser = argparse.ArgumentParser(
        description="Generate word-level text with a trained RNN-family model."
    )
    parser.add_argument(
        "--model",
        choices=["rnn", "birnn", "lstm", "gru", "all"],
        default="all",
    )
    parser.add_argument(
        "--seed-text",
        default="First Citizen: Before we proceed",
        help="Text prompt that starts the generation.",
    )
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--random-seed", type=int, default=SEED)

    args = parser.parse_args()

    if args.tokens < 1:
        parser.error("--tokens must be at least 1.")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero.")

    device = torch.device("cpu")
    token_to_id, id_to_token = load_vocabulary()

    architectures = (
        ["rnn", "birnn", "lstm", "gru"]
        if args.model == "all"
        else [args.model]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for architecture in architectures:
        checkpoint_path = MODEL_DIR / f"{architecture}_best.pt"

        if not checkpoint_path.exists():
            print(f"\nSkipping {architecture.upper()}: checkpoint not found.")
            continue

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        model = NextWordModel(
            architecture=checkpoint["architecture"],
            vocab_size=checkpoint["vocab_size"],
            embedding_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"],
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])

        generated_text = generate(
            model=model,
            prompt=args.seed_text,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            context_length=checkpoint["context_length"],
            num_tokens=args.tokens,
            temperature=args.temperature,
            device=device,
            seed=args.random_seed,
        )

        filename = (
            f"{architecture}_temp{args.temperature}_"
            f"seed{args.random_seed}.txt"
        )
        output_path = OUTPUT_DIR / filename
        output_path.write_text(generated_text, encoding="utf-8")

        print("\n" + "=" * 65)
        print(f"MODEL: {architecture.upper()}")
        print(f"TEMPERATURE: {args.temperature}")
        print(f"GENERATED TOKENS: {args.tokens}")
        print("=" * 65)
        print(generated_text)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
