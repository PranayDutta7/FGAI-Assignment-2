
from pathlib import Path
import json
import re
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
INPUT_FILE = DATA_DIR / "input.txt"
OUTPUT_DIR = DATA_DIR / "processed_word"

VALIDATION_FRACTION = 0.10
UNK_TOKEN = "<UNK>"


def tokenize(text):
    """
    Split text into word-level tokens.
    Preserve punctuation and contractions.
    Examples: "Romeo:" -> ["Romeo", ":"]
              "don't" -> ["don't"]
    """
    return re.findall(
        r"\w+(?:['’]\w+)*|[^\w\s]",
        text,
        flags=re.UNICODE
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    text = INPUT_FILE.read_text(encoding="utf-8")
    tokens = tokenize(text)

    if len(tokens) < 100:
        raise ValueError("Tokenized dataset is unexpectedly small.")

    # Split chronologically BEFORE building the vocabulary.
    split_index = int(len(tokens) * (1 - VALIDATION_FRACTION))
    train_tokens = tokens[:split_index]
    validation_tokens = tokens[split_index:]

    # Build vocabulary using training tokens only.
    vocab = [UNK_TOKEN] + sorted(set(train_tokens))
    word_to_id = {
        word: index for index, word in enumerate(vocab)
    }
    id_to_word = {
        str(index): word for index, word in enumerate(vocab)
    }

    # Encode words. Unseen validation words become <UNK>.
    train_ids = np.array(
        [word_to_id[word] for word in train_tokens],
        dtype=np.int32
    )
    validation_ids = np.array(
        [word_to_id.get(word, word_to_id[UNK_TOKEN])
         for word in validation_tokens],
        dtype=np.int32
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "train_ids.npy", train_ids)
    np.save(OUTPUT_DIR / "validation_ids.npy", validation_ids)

    with open(OUTPUT_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "word_to_id": word_to_id,
                "id_to_word": id_to_word
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    unknown_count = sum(
        1 for word in validation_tokens if word not in word_to_id
    )

    metadata = {
        "dataset": "Tiny Shakespeare",
        "level": "word",
        "tokenizer": "Unicode word tokens with punctuation",
        "total_tokens": len(tokens),
        "vocabulary_size": len(vocab),
        "training_tokens": len(train_tokens),
        "validation_tokens": len(validation_tokens),
        "validation_fraction": VALIDATION_FRACTION,
        "validation_unknown_tokens": unknown_count,
        "validation_unknown_fraction": (
            unknown_count / len(validation_tokens)
        ),
        "split_method": "chronological",
        "unknown_token": UNK_TOKEN
    }

    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Integrity checks
    assert len(train_ids) == len(train_tokens)
    assert len(validation_ids) == len(validation_tokens)
    assert len(train_ids) + len(validation_ids) == len(tokens)
    assert train_ids.min() >= 0
    assert validation_ids.max() < len(vocab)

    print("WORD-LEVEL PREPROCESSING COMPLETE")
    print(f"Total tokens: {len(tokens):,}")
    print(f"Training tokens: {len(train_ids):,}")
    print(f"Validation tokens: {len(validation_ids):,}")
    print(f"Vocabulary size: {len(vocab):,}")
    print(
        f"Unseen validation tokens: {unknown_count:,} "
        f"({100 * unknown_count / len(validation_tokens):.2f}%)"
    )
    print(f"Output directory: {OUTPUT_DIR}")

    sample = " ".join(
        vocab[int(i)] for i in train_ids[:40]
    )
    print("\nDecoded training sample:")
    print(sample)
    print("\nAll integrity checks passed!")


if __name__ == "__main__":
    main()
