"""
Phase 2: Neural Bigram Name Generator
=====================================

This file replaces the count matrix from Phase 1 with a tiny neural network.

The learning task stays the same:

    Given the current character, predict the next character.

Example:

    .emma.

Training pairs:

    . -> e
    e -> m
    m -> m
    m -> a
    a -> .

Architecture:

    Input Character ID
          |
          v
    Embedding Layer
          |
          v
    Linear Layer
          |
          v
    Softmax probabilities
          |
          v
    Next Character Prediction

Important PyTorch detail:
    During training, nn.CrossEntropyLoss expects raw logits, not softmax
    probabilities. It applies the softmax calculation internally in a stable
    way. During generation, we manually apply softmax so we can sample from
    probabilities.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


START_END_TOKEN = "."


@dataclass(frozen=True)
class Vocabulary:
    """Character vocabulary plus lookup dictionaries."""

    chars: list[str]
    char_to_idx: dict[str, int]
    idx_to_char: dict[int, str]

    @property
    def size(self) -> int:
        """Return number of tokens in the vocabulary."""
        return len(self.chars)


class NeuralBigramModel(nn.Module):
    """A neural network that predicts the next character from one character.

    What is an embedding?
        An embedding is a learned vector representation of a token. Instead of
        treating "a" as only the integer 1, the model learns a small vector for
        "a". Similar or useful characters can learn useful numeric patterns.

    Why embeddings are better than plain integer encoding?
        Integer ids are just labels. The id for "b" is larger than "a", but
        that does not mean "b" has more meaning than "a". Embeddings give each
        token a trainable vector, so the model can learn representations.

    Forward propagation:
        The forward method defines how data flows through the network:

            input ids -> embedding vectors -> linear layer -> logits
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run forward propagation.

        Input shape:
            (batch_size,)

        Embedding output shape:
            (batch_size, embedding_dim)

        Logits output shape:
            (batch_size, vocab_size)
        """
        embeddings = self.embedding(input_ids)
        logits = self.linear(embeddings)
        return logits


def load_names(data_path: Path) -> list[str]:
    """Load and preprocess the names dataset."""
    names = [
        line.strip().lower()
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not names:
        raise ValueError(f"No names found in {data_path}")

    return names


def build_vocabulary(names: list[str]) -> Vocabulary:
    """Build character vocabulary, char_to_idx, and idx_to_char mappings."""
    unique_chars = sorted(set("".join(names)))
    chars = [START_END_TOKEN] + unique_chars
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return Vocabulary(chars=chars, char_to_idx=char_to_idx, idx_to_char=idx_to_char)


def add_start_end_token(name: str) -> str:
    """Add the special boundary token to a name.

    Example:
        emma -> .emma.
    """
    return f"{START_END_TOKEN}{name}{START_END_TOKEN}"


def create_training_tensors(names: list[str], vocabulary: Vocabulary) -> tuple[torch.Tensor, torch.Tensor]:
    """Create all bigram training pairs and convert them to tensors.

    For ".emma.":

        input:  .  target: e
        input:  e  target: m
        input:  m  target: m
        input:  m  target: a
        input:  a  target: .

    xs shape:
        (number_of_pairs,)

    ys shape:
        (number_of_pairs,)
    """
    xs: list[int] = []
    ys: list[int] = []

    for name in names:
        tokenized_name = add_start_end_token(name)
        for current_char, next_char in zip(tokenized_name, tokenized_name[1:]):
            xs.append(vocabulary.char_to_idx[current_char])
            ys.append(vocabulary.char_to_idx[next_char])

    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def train_model(
    model: NeuralBigramModel,
    xs: torch.Tensor,
    ys: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> list[float]:
    """Train the neural bigram model and return loss history.

    Cross entropy loss:
        Measures how far the predicted next-character distribution is from the
        correct target character. Lower loss means the model assigns more
        probability to the correct next character.

    Backpropagation:
        After computing loss, loss.backward() calculates gradients. A gradient
        tells each weight how it should change to reduce the loss.

    Gradient descent:
        The optimizer updates weights in the direction that lowers loss.

    Adam optimizer:
        Adam is a popular optimizer that adapts the update size for each
        parameter using running averages of gradients. It often trains faster
        and more smoothly than plain gradient descent.
    """
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []

    print("\nTraining")
    print("-" * 60)
    print(f"Training pairs: {xs.shape[0]}")
    print(f"Input tensor shape:  {tuple(xs.shape)}")
    print(f"Target tensor shape: {tuple(ys.shape)}")

    for epoch in range(1, epochs + 1):
        batch_indices = torch.randint(low=0, high=xs.shape[0], size=(batch_size,))
        xb = xs[batch_indices]
        yb = ys[batch_indices]

        logits = model(xb)
        loss = loss_function(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch == 1 or epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch:5d}/{epochs} | loss = {loss.item():.4f}")

    return losses


def plot_loss_curve(losses: list[float], output_path: Path) -> None:
    """Save a training loss curve with matplotlib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(losses) + 1), losses, color="#2563eb", linewidth=1.5)
    ax.set_title("Neural Bigram Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross Entropy Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_name(
    model: NeuralBigramModel,
    vocabulary: Vocabulary,
    max_length: int = 20,
) -> str:
    """Generate one name using neural network probability sampling."""
    model.eval()
    current_idx = vocabulary.char_to_idx[START_END_TOKEN]
    generated_chars: list[str] = []

    with torch.no_grad():
        for _ in range(max_length):
            input_tensor = torch.tensor([current_idx], dtype=torch.long)
            logits = model(input_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0)
            next_idx = torch.multinomial(probabilities, num_samples=1).item()
            next_char = vocabulary.idx_to_char[next_idx]

            if next_char == START_END_TOKEN:
                break

            generated_chars.append(next_char)
            current_idx = next_idx

    return "".join(generated_chars)


def generate_names(
    model: NeuralBigramModel,
    vocabulary: Vocabulary,
    count: int,
    max_length: int = 20,
) -> list[str]:
    """Generate multiple non-empty names."""
    names: list[str] = []
    attempts = 0
    max_attempts = count * 20

    while len(names) < count and attempts < max_attempts:
        attempts += 1
        name = generate_name(model, vocabulary, max_length=max_length)
        if name:
            names.append(name)

    return names


def predict_next_characters(
    model: NeuralBigramModel,
    text: str,
    vocabulary: Vocabulary,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return top-k next-character probabilities for a user-provided prefix.

    Important limitation:
        This is a neural bigram model, so it only uses the last character of the
        input prefix. For input "em", the prediction is based on "m".
    """
    cleaned_text = text.strip().lower()
    if not cleaned_text:
        current_char = START_END_TOKEN
    else:
        current_char = cleaned_text[-1]

    if current_char not in vocabulary.char_to_idx:
        raise ValueError(f"Character '{current_char}' is not in the vocabulary.")

    current_idx = vocabulary.char_to_idx[current_char]

    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([current_idx], dtype=torch.long)
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)
        top_probabilities, top_indices = torch.topk(probabilities, k=top_k)

    predictions: list[tuple[str, float]] = []
    for probability, token_idx in zip(top_probabilities.tolist(), top_indices.tolist()):
        predictions.append((vocabulary.idx_to_char[token_idx], probability))

    return predictions


def save_generated_names(names: list[str], output_path: Path) -> None:
    """Save generated names to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(names) + "\n", encoding="utf-8")


def print_dataset_summary(names: list[str], vocabulary: Vocabulary) -> None:
    """Print dataset and vocabulary details."""
    longest_name = max(names, key=len)
    average_length = sum(len(name) for name in names) / len(names)

    print("Dataset Summary")
    print("-" * 60)
    print(f"Total names:         {len(names)}")
    print(f"Vocabulary size:     {vocabulary.size}")
    print(f"Longest name:        {longest_name} ({len(longest_name)} chars)")
    print(f"Average name length: {average_length:.2f}")
    print(f"Vocabulary:          {vocabulary.chars}")
    print(f"char_to_idx:         {vocabulary.char_to_idx}")


def print_training_examples(names: list[str], vocabulary: Vocabulary) -> None:
    """Print beginner-friendly examples of tokenization and pair creation."""
    example_name = names[0]
    tokenized_name = add_start_end_token(example_name)

    print("\nTraining Pair Example")
    print("-" * 60)
    print(f"Name: {example_name}")
    print(f"With start/end token: {tokenized_name}")

    for current_char, next_char in zip(tokenized_name, tokenized_name[1:]):
        current_idx = vocabulary.char_to_idx[current_char]
        next_idx = vocabulary.char_to_idx[next_char]
        print(f"{current_char} -> {next_char}    ({current_idx} -> {next_idx})")


def print_top_predictions(predictions: list[tuple[str, float]], prompt: str) -> None:
    """Display top predicted next characters."""
    last_char = prompt[-1].lower() if prompt else START_END_TOKEN
    print("\nTop Next-Character Predictions")
    print("-" * 60)
    print(f"Input text: '{prompt}'")
    print(f"Bigram context actually used: '{last_char}'")

    for rank, (char, probability) in enumerate(predictions, start=1):
        display_char = "<END>" if char == START_END_TOKEN else char
        print(f"{rank}. {display_char:5s} probability = {probability:.4f}")


def run_project(
    data_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    embedding_dim: int,
    learning_rate: float,
    generated_count: int,
    seed: int,
    prompt: str,
) -> None:
    """Run the full neural bigram pipeline."""
    random.seed(seed)
    torch.manual_seed(seed)

    names = load_names(data_path)
    vocabulary = build_vocabulary(names)
    xs, ys = create_training_tensors(names, vocabulary)

    model = NeuralBigramModel(vocab_size=vocabulary.size, embedding_dim=embedding_dim)

    print("Neural Bigram Name Generator")
    print("=" * 60)
    print_dataset_summary(names, vocabulary)
    print_training_examples(names, vocabulary)
    print("\nModel")
    print("-" * 60)
    print(model)

    with torch.no_grad():
        sample_logits = model(xs[:8])
        sample_embeddings = model.embedding(xs[:8])
        print("\nTensor Shapes")
        print("-" * 60)
        print(f"Sample input shape:      {tuple(xs[:8].shape)}")
        print(f"Sample embedding shape:  {tuple(sample_embeddings.shape)}")
        print(f"Sample logits shape:     {tuple(sample_logits.shape)}")

    losses = train_model(
        model=model,
        xs=xs,
        ys=ys,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    generated_names = generate_names(model, vocabulary, count=generated_count)
    predictions = predict_next_characters(model, prompt, vocabulary, top_k=5)

    generated_names_path = output_dir / "neural_generated_names.txt"
    loss_curve_path = output_dir / "neural_training_loss.png"
    save_generated_names(generated_names, generated_names_path)
    plot_loss_curve(losses, loss_curve_path)

    print("\nGenerated Names")
    print("-" * 60)
    for index, name in enumerate(generated_names, start=1):
        print(f"{index:2d}. {name.capitalize()}")

    print_top_predictions(predictions, prompt)

    print("\nSaved Outputs")
    print("-" * 60)
    print(f"Generated names: {generated_names_path}")
    print(f"Loss curve:      {loss_curve_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Train a Neural Bigram Name Generator.")
    parser.add_argument("--data", type=Path, default=project_root / "data" / "names.txt")
    parser.add_argument("--outputs", type=Path, default=project_root / "outputs")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        type=str,
        default="em",
        help="Prefix used for top-5 next-character prediction.",
    )
    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    args = parse_args()
    run_project(
        data_path=args.data,
        output_dir=args.outputs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        generated_count=args.count,
        seed=args.seed,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    main()
