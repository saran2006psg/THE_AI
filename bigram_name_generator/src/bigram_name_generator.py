"""
Bigram Name Generator
=====================

This project trains a simple character-level Bigram Language Model on a
dataset of names.

A bigram model learns probabilities like:

    P(next_character | current_character)

Example:

    . -> e
    e -> m
    m -> m
    m -> a
    a -> .

The "." token means both:
    1. Start of a name
    2. End of a name

This is a tiny version of next-token prediction, the same broad learning idea
behind modern language models. The difference is that GPT predicts the next
token using a long context and a Transformer, while this project predicts the
next character using only the previous character.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


START_END_TOKEN = "."


@dataclass(frozen=True)
class DatasetStats:
    """Summary statistics for the names dataset."""

    total_names: int
    vocabulary_size: int
    longest_name: str
    average_name_length: float


@dataclass(frozen=True)
class Vocabulary:
    """Character-level vocabulary and lookup dictionaries.

    char_to_idx:
        Converts a character into an integer index.

    idx_to_char:
        Converts an integer index back into a character.

    Why this matters:
        Computers and ML models work with numbers. Tokenization is the bridge
        between human text and numeric computation.
    """

    chars: list[str]
    char_to_idx: dict[str, int]
    idx_to_char: dict[int, str]

    @property
    def size(self) -> int:
        """Return the number of unique tokens."""
        return len(self.chars)


def load_names(data_path: Path) -> list[str]:
    """Load names from a text file.

    Expected file format:
        One name per line.

    Cleaning performed:
        - Strip surrounding whitespace
        - Convert to lowercase
        - Remove empty lines
    """
    names = [
        line.strip().lower()
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not names:
        raise ValueError(f"No names were found in {data_path}")

    return names


def build_vocabulary(names: list[str]) -> Vocabulary:
    """Build a sorted character vocabulary from all names.

    The start/end token is placed first, so it always has index 0.
    """
    unique_name_chars = sorted(set("".join(names)))
    chars = [START_END_TOKEN] + unique_name_chars
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    return Vocabulary(chars=chars, char_to_idx=char_to_idx, idx_to_char=idx_to_char)


def analyze_dataset(names: list[str], vocabulary: Vocabulary) -> DatasetStats:
    """Compute simple dataset statistics for learning and reporting."""
    longest_name = max(names, key=len)
    average_name_length = sum(len(name) for name in names) / len(names)

    return DatasetStats(
        total_names=len(names),
        vocabulary_size=vocabulary.size,
        longest_name=longest_name,
        average_name_length=average_name_length,
    )


def encode(text: str, vocabulary: Vocabulary) -> list[int]:
    """Convert a string into character token ids."""
    return [vocabulary.char_to_idx[char] for char in text]


def decode(token_ids: list[int], vocabulary: Vocabulary) -> str:
    """Convert character token ids back into a string."""
    return "".join(vocabulary.idx_to_char[token_id] for token_id in token_ids)


def add_start_end_tokens(name: str) -> str:
    """Add the special token to the beginning and end of a name.

    Example:
        emma -> .emma.
    """
    return f"{START_END_TOKEN}{name}{START_END_TOKEN}"


def create_training_pairs(names: list[str], vocabulary: Vocabulary) -> list[tuple[int, int]]:
    """Create current-character -> next-character training pairs.

    For the name "emma":

        .emma.

    The pairs are:

        . -> e
        e -> m
        m -> m
        m -> a
        a -> .

    Each pair is stored as integer ids instead of raw characters.
    """
    pairs: list[tuple[int, int]] = []

    for name in names:
        tokenized_name = add_start_end_tokens(name)

        for current_char, next_char in zip(tokenized_name, tokenized_name[1:]):
            current_idx = vocabulary.char_to_idx[current_char]
            next_idx = vocabulary.char_to_idx[next_char]
            pairs.append((current_idx, next_idx))

    return pairs


def build_bigram_count_matrix(
    training_pairs: list[tuple[int, int]],
    vocabulary_size: int,
) -> list[list[int]]:
    """Build a square count matrix for character transitions.

    Matrix shape:
        (vocabulary_size, vocabulary_size)

    Meaning:
        matrix[i][j] = how many times token i is followed by token j
    """
    counts = [[0 for _ in range(vocabulary_size)] for _ in range(vocabulary_size)]

    for current_idx, next_idx in training_pairs:
        counts[current_idx][next_idx] += 1

    return counts


def counts_to_probabilities(counts: list[list[int]], smoothing: float = 1.0) -> list[list[float]]:
    """Convert raw counts into row-wise probabilities.

    Formula:

        P(next = j | current = i) =
            count(i, j) / sum_over_all_next_chars(count(i, next))

    This implementation uses add-one smoothing:

        (count(i, j) + 1) / (row_sum + vocabulary_size)

    Smoothing prevents zero-probability transitions. That matters because a
    transition that never appeared in training can still be sampled rarely.
    """
    probabilities: list[list[float]] = []

    for row in counts:
        smoothed_row = [count + smoothing for count in row]
        row_total = sum(smoothed_row)
        probabilities.append([value / row_total for value in smoothed_row])

    return probabilities


def sample_next_index(probabilities: list[float]) -> int:
    """Sample one index from a probability distribution."""
    return random.choices(
        population=range(len(probabilities)),
        weights=probabilities,
        k=1,
    )[0]


def generate_name(
    probabilities: list[list[float]],
    vocabulary: Vocabulary,
    max_length: int = 20,
) -> str:
    """Generate one name from the bigram probability matrix.

    Generation loop:

        Start with "."
            |
            v
        sample next character
            |
            v
        stop if "." appears again

    The model only knows the current character. It does not remember the full
    name prefix, which is the core limitation of bigram models.
    """
    current_idx = vocabulary.char_to_idx[START_END_TOKEN]
    generated_chars: list[str] = []

    for _ in range(max_length):
        next_idx = sample_next_index(probabilities[current_idx])
        next_char = vocabulary.idx_to_char[next_idx]

        if next_char == START_END_TOKEN:
            break

        generated_chars.append(next_char)
        current_idx = next_idx

    return "".join(generated_chars)


def generate_names(
    probabilities: list[list[float]],
    vocabulary: Vocabulary,
    count: int,
    max_length: int = 20,
) -> list[str]:
    """Generate multiple non-empty names."""
    generated_names: list[str] = []
    attempts = 0
    max_attempts = count * 20

    while len(generated_names) < count and attempts < max_attempts:
        attempts += 1
        name = generate_name(probabilities, vocabulary, max_length=max_length)
        if name:
            generated_names.append(name)

    return generated_names


def save_generated_names(generated_names: list[str], output_path: Path) -> None:
    """Save generated names to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(generated_names) + "\n", encoding="utf-8")


def save_bigram_heatmap(
    probabilities: list[list[float]],
    vocabulary: Vocabulary,
    output_path: Path,
) -> None:
    """Visualize the bigram probability matrix as a heatmap.

    Brighter cells mean higher probability.

    Example:
        If the cell row "e", column "m" is bright, the model often predicts
        "m" after seeing "e".
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(probabilities, cmap="viridis")

    ax.set_title("Bigram Probability Matrix", fontsize=16, pad=16)
    ax.set_xlabel("Next Character")
    ax.set_ylabel("Current Character")

    ax.set_xticks(range(vocabulary.size))
    ax.set_yticks(range(vocabulary.size))
    ax.set_xticklabels(vocabulary.chars)
    ax.set_yticklabels(vocabulary.chars)

    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.colorbar(image, ax=ax, label="Probability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_dataset_report(stats: DatasetStats) -> None:
    """Print dataset statistics."""
    print("\nDataset Analysis")
    print("-" * 60)
    print(f"Total names:         {stats.total_names}")
    print(f"Vocabulary size:     {stats.vocabulary_size}")
    print(f"Longest name:        {stats.longest_name} ({len(stats.longest_name)} chars)")
    print(f"Average name length: {stats.average_name_length:.2f}")


def print_tokenization_examples(names: list[str], vocabulary: Vocabulary) -> None:
    """Print examples that explain character-level tokenization."""
    print("\nTokenization Examples")
    print("-" * 60)
    print(f"Vocabulary: {vocabulary.chars}")
    print(f"char_to_idx: {vocabulary.char_to_idx}")

    for name in names[:5]:
        encoded = encode(name, vocabulary)
        decoded = decode(encoded, vocabulary)
        print(f"{name:12s} -> {encoded} -> {decoded}")


def print_training_pair_examples(names: list[str], vocabulary: Vocabulary) -> None:
    """Print current-character to next-character examples."""
    print("\nTraining Pair Examples")
    print("-" * 60)

    example_name = names[0]
    tokenized_name = add_start_end_tokens(example_name)
    print(f"Original name: {example_name}")
    print(f"With start/end token: {tokenized_name}")

    for current_char, next_char in zip(tokenized_name, tokenized_name[1:]):
        current_idx = vocabulary.char_to_idx[current_char]
        next_idx = vocabulary.char_to_idx[next_char]
        print(f"{current_char} -> {next_char}    ({current_idx} -> {next_idx})")


def print_count_matrix_preview(counts: list[list[int]], vocabulary: Vocabulary, rows: int = 8) -> None:
    """Print a small preview of the count matrix."""
    print("\nBigram Count Matrix Preview")
    print("-" * 60)
    shown_chars = vocabulary.chars[:rows]
    header = "     " + " ".join(f"{char:>5s}" for char in shown_chars)
    print(header)

    for row_idx, row_char in enumerate(shown_chars):
        values = " ".join(f"{counts[row_idx][col_idx]:5d}" for col_idx in range(rows))
        print(f"{row_char:>3s}: {values}")


def print_probability_preview(probabilities: list[list[float]], vocabulary: Vocabulary, rows: int = 8) -> None:
    """Print a small preview of the probability matrix."""
    print("\nBigram Probability Matrix Preview")
    print("-" * 60)
    shown_chars = vocabulary.chars[:rows]
    header = "     " + " ".join(f"{char:>7s}" for char in shown_chars)
    print(header)

    for row_idx, row_char in enumerate(shown_chars):
        values = " ".join(f"{probabilities[row_idx][col_idx]:7.3f}" for col_idx in range(rows))
        print(f"{row_char:>3s}: {values}")


def run_project(
    data_path: Path,
    output_dir: Path,
    generated_count: int,
    seed: int,
) -> None:
    """Run the full Bigram Name Generator pipeline."""
    random.seed(seed)

    print("Bigram Name Generator")
    print("=" * 60)

    names = load_names(data_path)
    vocabulary = build_vocabulary(names)
    stats = analyze_dataset(names, vocabulary)
    training_pairs = create_training_pairs(names, vocabulary)
    counts = build_bigram_count_matrix(training_pairs, vocabulary.size)
    probabilities = counts_to_probabilities(counts)

    generated_names = generate_names(
        probabilities=probabilities,
        vocabulary=vocabulary,
        count=generated_count,
    )

    generated_names_path = output_dir / "generated_names.txt"
    heatmap_path = output_dir / "bigram_heatmap.png"

    save_generated_names(generated_names, generated_names_path)
    save_bigram_heatmap(probabilities, vocabulary, heatmap_path)

    print_dataset_report(stats)
    print_tokenization_examples(names, vocabulary)
    print_training_pair_examples(names, vocabulary)
    print_count_matrix_preview(counts, vocabulary)
    print_probability_preview(probabilities, vocabulary)

    print("\nGenerated Names")
    print("-" * 60)
    for index, name in enumerate(generated_names, start=1):
        print(f"{index:2d}. {name.capitalize()}")

    print("\nSaved Outputs")
    print("-" * 60)
    print(f"Generated names: {generated_names_path}")
    print(f"Heatmap:         {heatmap_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Train and run a Bigram Name Generator.")
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "data" / "names.txt",
        help="Path to names.txt with one name per line.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=project_root / "outputs",
        help="Directory where generated names and heatmap are saved.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of names to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    args = parse_args()
    run_project(
        data_path=args.data,
        output_dir=args.outputs,
        generated_count=args.count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
