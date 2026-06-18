"""
Tiny Language Model and Name Generator
======================================

This project is written as a learning notebook-style Python script.

Goal:
    Build a tiny character-level language model in multiple phases:

    Phase 1: Dataset and tokenization
    Phase 2: Frequency-based bigram language model
    Phase 3: Neural network bigram model with PyTorch
    Phase 4: Word generation from a starting character
    Phase 5: Name generation with probability scores

The model learns this simple task:

    Given one character, predict the next character.

That is called next token prediction. GPT-style models also learn by predicting
the next token, but they use subword tokens, huge datasets, and Transformer
attention layers instead of this tiny bigram architecture.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# A special token used for both the start and end of a name.
#
# Example training word:
#
#     emma
#
# We train these pairs:
#
#     . -> e
#     e -> m
#     m -> m
#     m -> a
#     a -> .
#
# The final "." teaches the model when to stop generating.
SPECIAL_TOKEN = "."


def section(title: str) -> None:
    """Print a readable section title."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def explain(text: str) -> None:
    """Print educational text with consistent spacing."""
    print("\n" + text.strip() + "\n")


@dataclass
class Vocabulary:
    """A small object that keeps token lookup tables together.

    char_to_idx:
        Dictionary that maps a character token to an integer id.
        Example: {"." : 0, "a": 1, "b": 2}

    idx_to_char:
        Dictionary that maps an integer id back to a character token.
        Example: {0: ".", 1: "a", 2: "b"}

    Why integer ids?
        Neural networks do not directly understand text. They work with
        numbers, so tokenization converts text into integer ids.
    """

    chars: list[str]
    char_to_idx: dict[str, int]
    idx_to_char: dict[int, str]

    @property
    def size(self) -> int:
        return len(self.chars)


class NeuralBigramLanguageModel(nn.Module):
    """A neural network that predicts the next character from one character.

    Diagram:

        Input character id
              |
              v
        Embedding layer
              |
              v
        Linear layer
              |
              v
        Logits for every possible next character
              |
              v
        Softmax probabilities during generation

    Input shape:
        (batch_size,)

    Embedding output shape:
        (batch_size, embedding_dim)

    Linear output shape:
        (batch_size, vocab_size)

    The linear output is called logits. CrossEntropyLoss expects raw logits,
    so we do not apply softmax before the loss.
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        logits = self.linear(embeddings)
        return logits


def load_names(path: Path) -> list[str]:
    """Load a text file where every line contains one name."""
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip().lower()
        if name:
            names.append(name)

    if not names:
        raise ValueError(f"No names found in {path}")

    return names


def build_vocabulary(names: list[str]) -> Vocabulary:
    """Build a character vocabulary from the dataset."""
    unique_chars = sorted(set("".join(names)))

    # Put the special token first so its id is always 0.
    chars = [SPECIAL_TOKEN] + unique_chars
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    return Vocabulary(chars=chars, char_to_idx=char_to_idx, idx_to_char=idx_to_char)


def encode(text: str, vocab: Vocabulary) -> list[int]:
    """Convert characters into integer token ids."""
    return [vocab.char_to_idx[char] for char in text]


def decode(token_ids: list[int], vocab: Vocabulary) -> str:
    """Convert integer token ids back into characters."""
    return "".join(vocab.idx_to_char[token_id] for token_id in token_ids)


def make_bigram_pairs(names: list[str], vocab: Vocabulary) -> tuple[torch.Tensor, torch.Tensor]:
    """Create supervised learning examples for next-character prediction.

    For each name, add a start token and an end token:

        "emma" becomes ".emma."

    Then create input/target pairs:

        input:  "."  target: "e"
        input:  "e"  target: "m"
        input:  "m"  target: "m"
        input:  "m"  target: "a"
        input:  "a"  target: "."

    Input shape:
        xs has shape (number_of_pairs,)

    Output shape:
        ys has shape (number_of_pairs,)
    """
    xs: list[int] = []
    ys: list[int] = []

    for name in names:
        tokens = SPECIAL_TOKEN + name + SPECIAL_TOKEN
        for current_char, next_char in zip(tokens, tokens[1:]):
            xs.append(vocab.char_to_idx[current_char])
            ys.append(vocab.char_to_idx[next_char])

    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def phase_1_dataset_and_tokenization(names: list[str], vocab: Vocabulary) -> None:
    section("PHASE 1 - DATASET AND TOKENIZATION")

    explain(
        """
        Problem solved:
            Raw text cannot be fed directly into a neural network.
            Tokenization converts text into numbers.

        In this project:
            Token = one character
            Vocabulary = all characters the model knows
            Encoding = characters -> integer ids
            Decoding = integer ids -> characters

        Relation to modern LLMs:
            GPT models also tokenize text. They usually use subword tokens
            instead of single characters, but the idea is the same: text becomes
            integer ids before entering the model.
        """
    )

    print("First 10 names from dataset:")
    print(names[:10])

    print("\nVocabulary:")
    print(vocab.chars)

    print("\nVocabulary size:")
    print(vocab.size)

    print("\nchar_to_idx dictionary:")
    print(vocab.char_to_idx)

    print("\nidx_to_char dictionary:")
    print(vocab.idx_to_char)

    examples = names[:5]
    print("\nEncoded and decoded examples:")
    for name in examples:
        encoded = encode(name, vocab)
        decoded = decode(encoded, vocab)
        print(f"{name:10s} -> {encoded} -> {decoded}")

    explain(
        """
        Input shape:
            A single name becomes a Python list with length equal to the number
            of characters in the name.

        Example:
            "emma" has 4 characters, so its encoded shape is conceptually (4,).

        Why needed:
            Every later phase depends on this mapping from symbols to numbers.
        """
    )


def build_bigram_count_matrix(names: list[str], vocab: Vocabulary) -> torch.Tensor:
    """Count how often each character is followed by each other character."""
    counts = torch.zeros((vocab.size, vocab.size), dtype=torch.int32)

    for name in names:
        tokens = SPECIAL_TOKEN + name + SPECIAL_TOKEN
        for current_char, next_char in zip(tokens, tokens[1:]):
            current_idx = vocab.char_to_idx[current_char]
            next_idx = vocab.char_to_idx[next_char]
            counts[current_idx, next_idx] += 1

    return counts


def counts_to_probabilities(counts: torch.Tensor) -> torch.Tensor:
    """Convert counts into probabilities.

    We add 1 to every count. This is called smoothing.

    Why smoothing?
        If a bigram never appears in the training data, its count is zero.
        Without smoothing, that transition would have probability zero and
        could never be sampled.
    """
    smoothed_counts = counts.float() + 1.0
    row_sums = smoothed_counts.sum(dim=1, keepdim=True)
    probabilities = smoothed_counts / row_sums
    return probabilities


def sample_from_probability_row(probabilities: torch.Tensor) -> tuple[int, float]:
    """Sample one token id from a probability distribution."""
    sampled_idx = torch.multinomial(probabilities, num_samples=1, replacement=True).item()
    sampled_probability = probabilities[sampled_idx].item()
    return sampled_idx, sampled_probability


def generate_with_frequency_bigram(
    probabilities: torch.Tensor,
    vocab: Vocabulary,
    start_char: str | None = None,
    max_length: int = 20,
) -> tuple[str, float]:
    """Generate a word using the frequency-based bigram probabilities."""
    if start_char is None:
        current_idx = vocab.char_to_idx[SPECIAL_TOKEN]
        generated_chars: list[str] = []
    else:
        current_idx = vocab.char_to_idx[start_char]
        generated_chars = [start_char]

    total_log_probability = 0.0

    for _ in range(max_length):
        next_idx, probability = sample_from_probability_row(probabilities[current_idx])
        next_char = vocab.idx_to_char[next_idx]
        total_log_probability += math.log(probability)

        if next_char == SPECIAL_TOKEN:
            break

        generated_chars.append(next_char)
        current_idx = next_idx

    probability_score = math.exp(total_log_probability)
    return "".join(generated_chars), probability_score


def print_matrix(matrix: torch.Tensor, vocab: Vocabulary, title: str, decimals: int | None = None) -> None:
    """Pretty-print a small matrix with character labels."""
    print(f"\n{title}")
    header = "     " + " ".join(f"{char:>6s}" for char in vocab.chars)
    print(header)

    for row_idx, row_char in enumerate(vocab.chars):
        values = []
        for col_idx in range(vocab.size):
            value = matrix[row_idx, col_idx].item()
            if decimals is None:
                values.append(f"{int(value):6d}")
            else:
                values.append(f"{value:6.{decimals}f}")
        print(f"{row_char:>3s}: " + " ".join(values))


def phase_2_frequency_bigram(names: list[str], vocab: Vocabulary) -> torch.Tensor:
    section("PHASE 2 - BIGRAM LANGUAGE MODEL")

    explain(
        """
        What is a bigram?
            A bigram is a pair of neighboring tokens.

            For "hello":
                h -> e
                e -> l
                l -> l
                l -> o

        What is a language model?
            A language model assigns probabilities to token sequences.
            In this tiny project, it learns:

                P(next character | current character)

        Problem solved:
            The model learns which characters usually follow other characters.

        Input shape:
            One current character id.

        Output shape:
            A probability distribution over all possible next characters.

        Relation to modern LLMs:
            GPT predicts the next token too, but it conditions on many previous
            tokens, not just one previous character.
        """
    )

    counts = build_bigram_count_matrix(names, vocab)
    probabilities = counts_to_probabilities(counts)

    print_matrix(counts, vocab, "Bigram count matrix:")
    print_matrix(probabilities, vocab, "Bigram probability matrix:", decimals=3)

    explain(
        """
        How probability-based generation works:
            1. Start at the special token "."
            2. Look up the row of probabilities for the current character
            3. Randomly sample the next character from that row
            4. Repeat until "." is sampled as the end token

        Sampling means we do not always pick the most likely character.
        This gives variety, but it can also create strange names.
        """
    )

    print("Example generations from frequency bigram model:")
    for _ in range(10):
        generated, score = generate_with_frequency_bigram(probabilities, vocab)
        print(f"{generated:12s} probability_score={score:.3e}")

    return probabilities


def phase_3_neural_bigram(
    names: list[str],
    vocab: Vocabulary,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> NeuralBigramLanguageModel:
    section("PHASE 3 - NEURAL NETWORK BIGRAM MODEL")

    explain(
        """
        We now replace the count table with a neural network.

        Architecture:

            Character id
                |
                v
            Embedding Layer
                |
                v
            Linear Layer
                |
                v
            Output logits
                |
                v
            Softmax probabilities during generation

        Problem solved:
            Instead of storing observed frequencies directly, the neural network
            learns parameters that predict the next character.

        Relation to modern LLMs:
            Modern LLMs also start with token embeddings and train with
            next-token prediction. Their middle layers are much more powerful
            because they use attention and deep Transformer blocks.
        """
    )

    xs, ys = make_bigram_pairs(names, vocab)

    print("Training data tensors:")
    print(f"xs shape: {tuple(xs.shape)}  (input character ids)")
    print(f"ys shape: {tuple(ys.shape)}  (target next-character ids)")
    print("\nSample training pairs:")
    for i in range(12):
        x_char = vocab.idx_to_char[xs[i].item()]
        y_char = vocab.idx_to_char[ys[i].item()]
        print(f"{xs[i].item():2d} ({x_char}) -> {ys[i].item():2d} ({y_char})")

    model = NeuralBigramLanguageModel(vocab_size=vocab.size, embedding_dim=embedding_dim)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("\nModel:")
    print(model)

    with torch.no_grad():
        sample_logits = model(xs[:batch_size])
        sample_embeddings = model.embedding(xs[:batch_size])
        print("\nTensor shapes through the model:")
        print(f"Input batch shape:       {tuple(xs[:batch_size].shape)}")
        print(f"Embedding batch shape:   {tuple(sample_embeddings.shape)}")
        print(f"Linear/logits shape:     {tuple(sample_logits.shape)}")

    explain(
        """
        Key concepts:

        Embeddings:
            A learned vector for each token id. Instead of representing "a" as
            only an integer, the model learns a small vector for "a".

        Forward propagation:
            Data moves from input -> embedding -> linear layer -> logits.

        Cross entropy loss:
            Measures how wrong the predicted next-character distribution is.
            Lower loss means the model assigns higher probability to the
            correct next character.

        Backpropagation:
            Computes how each parameter contributed to the loss.

        Gradient descent:
            Updates parameters in the direction that reduces loss.

        Adam optimizer:
            A popular gradient-based optimizer that adapts learning rates for
            parameters. It is widely used in deep learning.
        """
    )

    for epoch in range(1, epochs + 1):
        batch_indices = torch.randint(low=0, high=xs.shape[0], size=(batch_size,))
        xb = xs[batch_indices]
        yb = ys[batch_indices]

        logits = model(xb)
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch:4d}/{epochs} | loss = {loss.item():.4f}")

    return model


def generate_with_neural_model(
    model: NeuralBigramLanguageModel,
    vocab: Vocabulary,
    start_char: str | None = None,
    max_length: int = 20,
) -> tuple[str, float]:
    """Generate a word with the trained neural bigram model."""
    model.eval()

    if start_char is None:
        current_idx = vocab.char_to_idx[SPECIAL_TOKEN]
        generated_chars: list[str] = []
    else:
        current_idx = vocab.char_to_idx[start_char]
        generated_chars = [start_char]

    total_log_probability = 0.0

    with torch.no_grad():
        for _ in range(max_length):
            x = torch.tensor([current_idx], dtype=torch.long)
            logits = model(x)
            probabilities = F.softmax(logits, dim=1).squeeze(0)

            next_idx, probability = sample_from_probability_row(probabilities)
            next_char = vocab.idx_to_char[next_idx]
            total_log_probability += math.log(probability)

            if next_char == SPECIAL_TOKEN:
                break

            generated_chars.append(next_char)
            current_idx = next_idx

    probability_score = math.exp(total_log_probability)
    return "".join(generated_chars), probability_score


def phase_4_word_generation(
    model: NeuralBigramLanguageModel,
    vocab: Vocabulary,
    interactive: bool,
) -> None:
    section("PHASE 4 - WORD GENERATION")

    explain(
        """
        Problem solved:
            After training, we use the model to create new words one character
            at a time.

        Input shape:
            One current character id with shape (1,).

        Output shape:
            One probability distribution with shape (1, vocab_size).

        Sampling:
            The model outputs a probability for every possible next character.
            We sample from that distribution to choose the next character.

        Why generated words are sometimes incorrect:
            This model only sees one previous character. It has no memory of the
            full word so far. If it generated "alexa", it still only conditions
            on the last "a" when choosing the next character.

        Relation to modern LLMs:
            GPT also samples from next-token probabilities, but it uses a long
            context window and attention to condition on many tokens.
        """
    )

    valid_start_chars = [char for char in vocab.chars if char != SPECIAL_TOKEN]

    if interactive:
        user_char = input("Enter a starting character, or press Enter for random start: ").strip().lower()
        if user_char and user_char not in vocab.char_to_idx:
            print(f"'{user_char}' is not in the vocabulary. Using random start instead.")
            user_char = ""
        start_char = user_char[0] if user_char else random.choice(valid_start_chars)
    else:
        start_char = random.choice(valid_start_chars)
        print(f"Non-interactive mode: using random starting character '{start_char}'.")

    print(f"\nGenerated words starting with '{start_char}':")
    for _ in range(5):
        generated, score = generate_with_neural_model(model, vocab, start_char=start_char)
        print(f"{generated:12s} probability_score={score:.3e}")


def phase_5_name_generator(model: NeuralBigramLanguageModel, vocab: Vocabulary) -> None:
    section("PHASE 5 - NAME GENERATOR")

    explain(
        """
        Problem solved:
            Use the trained model as a name generator.

        Input shape:
            Generation begins from the special start token ".".

        Output shape:
            At each step, the model outputs probabilities for the next
            character. The final output is a variable-length generated name.

        Why generated names resemble training data:
            The model learned character transition patterns from real names.
            For example, it may learn that "a" often appears at the end of a
            name, or that "ch" is a common transition.

        Why they are not exact copies:
            Sampling mixes learned transitions in new ways. The model is not
            storing a list of names; it is predicting one next character at a
            time from learned probabilities.

        Relation to modern LLMs:
            Larger LLMs generate text by repeatedly predicting and sampling the
            next token. This phase is the same loop at a tiny character level.
        """
    )

    print("10 generated names:")
    generated_count = 0
    attempts = 0

    # The model can sometimes sample the end token immediately, creating an
    # empty string. That is valid model behavior, but not useful for displaying
    # "10 new names", so we retry empty generations.
    while generated_count < 10 and attempts < 100:
        attempts += 1
        generated, score = generate_with_neural_model(model, vocab)
        if not generated:
            continue

        display_name = generated.capitalize() if generated else "(empty)"
        print(f"{display_name:12s} probability_score={score:.3e}")
        generated_count += 1


def final_explanation() -> None:
    section("FINAL EXPLANATION")

    explain(
        """
        1. What is a Language Model?
            A language model predicts probabilities for text. In this project,
            it predicts the next character in a name. In GPT, the model predicts
            the next token in a much larger piece of text.

        2. What is Next Token Prediction?
            Next token prediction means:

                Given previous token(s), predict the next token.

            Tiny example:

                e m m -> predict a

            GPT example:

                The capital of France is -> predict Paris

        3. How GPT is similar to this project?
            Both learn from examples of real text.
            Both convert tokens into ids.
            Both use embeddings.
            Both output probabilities over a vocabulary.
            Both generate by repeatedly sampling the next token.

        4. Limitations of Bigram Models
            A bigram model only looks at one previous token. It cannot remember
            the whole prefix. It cannot understand long-range structure.

            Example:
                If the current character is "a", the model does not know
                whether the current partial name is "emma", "olivia", or "ava".

        5. Why Attention was invented
            Attention lets a model look back at many previous tokens and decide
            which ones matter for the next prediction.

        6. Why Transformers outperform Bigram Models
            Transformers use attention, many layers, nonlinear transformations,
            and huge training data. They can learn context, grammar, facts,
            style, and long-range dependencies far beyond one-token patterns.

        7. What Q, K, and V solve that this project cannot
            Q, K, and V mean Query, Key, and Value.

            Query:
                What is the current token looking for?

            Key:
                What information does each previous token offer?

            Value:
                What content should be passed forward if that token is useful?

            In attention, queries compare with keys to decide which values to
            use. This lets Transformers dynamically focus on relevant context.

            Our bigram model cannot do that. It only asks:

                What usually comes after this one character?

            A Transformer can ask:

                Which earlier tokens matter most for predicting the next token?
        """
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny Language Model and Name Generator")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data" / "names.txt",
        help="Path to a text file with one name per line.",
    )
    parser.add_argument("--epochs", type=int, default=500, help="Training steps for the neural model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--embedding-dim", type=int, default=16, help="Size of character embeddings.")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Adam learning rate.")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip the input prompt and choose a random starting character.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible learning runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    names = load_names(args.data)
    vocab = build_vocabulary(names)

    phase_1_dataset_and_tokenization(names, vocab)
    phase_2_frequency_bigram(names, vocab)
    model = phase_3_neural_bigram(
        names=names,
        vocab=vocab,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    phase_4_word_generation(model, vocab, interactive=not args.no_interactive)
    phase_5_name_generator(model, vocab)
    final_explanation()


if __name__ == "__main__":
    main()
