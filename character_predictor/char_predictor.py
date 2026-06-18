"""
Character Predictor from scratch using Python and PyTorch.

Project goal:
Given a sequence of characters, predict the next character.

Example:
Input:  "hell"
Output: "o"

This script is intentionally beginner-friendly and educational. It prints
intermediate outputs so you can see how raw text becomes tensors, how the
model is trained, and how predictions are made.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Configuration
# -----------------------------

DATA_FILE = Path(__file__).parent / "data.txt"
MAX_CONTEXT_LENGTH = 8
EMBEDDING_DIM = 16
HIDDEN_DIM = 64
EPOCHS = 250
LEARNING_RATE = 0.01
PRINT_EVERY = 25


def load_text(file_path):
    """
    Load a small text dataset from a text file.

    In real LLM training, the dataset can contain billions or trillions of
    tokens. Here we use a tiny text file so every step is easy to inspect.
    """
    text = file_path.read_text(encoding="utf-8").lower()
    return text


def build_vocabulary(text):
    """
    Create a vocabulary of unique characters.

    A vocabulary is the full set of tokens the model knows about.
    Because this is a character-level model, each token is one character.
    """
    vocab = sorted(set(text))

    # char_to_idx converts a character into an integer ID.
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}

    # idx_to_char converts an integer ID back into a character.
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    return vocab, char_to_idx, idx_to_char


def encode_text(text, char_to_idx):
    """
    Convert every character in the text into its integer ID.

    Neural networks cannot directly process raw characters like "h" or "e".
    They need numbers, so we encode characters as integer IDs.
    """
    encoded = [char_to_idx[char] for char in text]
    return encoded


def create_training_samples(encoded_text, max_context_length):
    """
    Generate training samples using a sliding window.

    For text "hello", examples include:
    "h"    -> "e"
    "he"   -> "l"
    "hel"  -> "l"
    "hell" -> "o"

    This function creates many examples from the full dataset. Each input can
    have a different length up to max_context_length. Before putting examples
    into tensors, we left-pad shorter inputs with 0 so every input has the same
    length.
    """
    inputs = []
    targets = []

    for target_position in range(1, len(encoded_text)):
        start_position = max(0, target_position - max_context_length)

        # The context is the sequence of previous characters.
        context = encoded_text[start_position:target_position]

        # The target is the next character after the context.
        target = encoded_text[target_position]

        # Pad shorter contexts on the left so all inputs have equal length.
        padding_needed = max_context_length - len(context)
        padded_context = [0] * padding_needed + context

        inputs.append(padded_context)
        targets.append(target)

    x = torch.tensor(inputs, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)

    return x, y


class CharacterPredictor(nn.Module):
    """
    A simple neural network for next-character prediction.

    Architecture:
    1. Embedding layer:
       Converts each character ID into a small learned vector.

    2. Linear hidden layer:
       Learns patterns from the flattened sequence of embeddings.

    3. Output layer:
       Produces one score, called a logit, for every character in the vocab.
    """

    def __init__(self, vocab_size, max_context_length, embedding_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # After embedding, input shape is:
        # [batch_size, max_context_length, embedding_dim]
        #
        # We flatten it into:
        # [batch_size, max_context_length * embedding_dim]
        self.hidden = nn.Linear(max_context_length * embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        Forward pass: turn input character IDs into prediction scores.
        """
        embedded = self.embedding(x)
        flattened = embedded.view(embedded.size(0), -1)
        hidden_output = torch.relu(self.hidden(flattened))
        logits = self.output(hidden_output)
        return logits


def print_learning_outputs(text, vocab, char_to_idx, idx_to_char, encoded_text, x, y):
    """
    Print intermediate outputs to make the pipeline easy to understand.
    """
    print("\n========== RAW TEXT ==========")
    print(text)

    print("\n========== VOCABULARY ==========")
    print(f"Vocabulary size: {len(vocab)}")
    print(vocab)

    print("\n========== char_to_idx ==========")
    print(char_to_idx)

    print("\n========== idx_to_char ==========")
    print(idx_to_char)

    print("\n========== ENCODED TEXT ==========")
    print("First 80 encoded character IDs:")
    print(encoded_text[:80])

    print("\n========== TRAINING SAMPLES ==========")
    print("First 10 samples as text:")
    for i in range(min(10, len(x))):
        input_ids = x[i].tolist()
        target_id = y[i].item()

        # Remove left-padding only for display.
        visible_input_ids = input_ids
        while len(visible_input_ids) > 1 and visible_input_ids[0] == 0:
            visible_input_ids = visible_input_ids[1:]

        input_text = "".join(idx_to_char[idx] for idx in visible_input_ids)
        target_text = idx_to_char[target_id]
        print(f"{i + 1:02d}. Input: {input_text!r:<12} -> Target: {target_text!r}")

    print("\n========== TENSOR SHAPES ==========")
    print(f"Input tensor x shape:  {x.shape}")
    print(f"Target tensor y shape: {y.shape}")
    print("x shape means: [number_of_samples, max_context_length]")
    print("y shape means: [number_of_samples]")


def train_model(model, x, y):
    """
    Train the model using CrossEntropyLoss and Adam.

    CrossEntropyLoss compares the model's predicted scores with the correct
    target character ID. Adam updates the model weights to reduce that loss.
    """
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n========== TRAINING ==========")
    for epoch in range(1, EPOCHS + 1):
        # Set gradients from the previous step to zero.
        optimizer.zero_grad()

        # Run the forward pass and get raw prediction scores.
        logits = model(x)

        # Calculate how wrong the predictions are.
        loss = loss_function(logits, y)

        # Backpropagation: calculate gradients for each trainable weight.
        loss.backward()

        # Optimizer step: update weights using the gradients.
        optimizer.step()

        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"Epoch {epoch:03d}/{EPOCHS} - Loss: {loss.item():.4f}")


def prepare_input_text(user_text, char_to_idx, max_context_length):
    """
    Encode user input and pad/truncate it to the model's expected length.

    If the user types more than max_context_length characters, we keep only
    the most recent characters because they are closest to the next prediction.
    """
    user_text = user_text.lower()

    unknown_chars = [char for char in user_text if char not in char_to_idx]
    if unknown_chars:
        raise ValueError(
            f"Unknown character(s): {sorted(set(unknown_chars))}. "
            "Try using characters that appear in the training text."
        )

    encoded = [char_to_idx[char] for char in user_text]
    encoded = encoded[-max_context_length:]

    padding_needed = max_context_length - len(encoded)
    padded = [0] * padding_needed + encoded

    return torch.tensor([padded], dtype=torch.long)


def predict_next_character(model, user_text, char_to_idx, idx_to_char):
    """
    Predict the next character and show the top 5 probabilities.
    """
    model.eval()

    input_tensor = prepare_input_text(user_text, char_to_idx, MAX_CONTEXT_LENGTH)

    with torch.no_grad():
        logits = model(input_tensor)

        # Softmax converts raw logits into probabilities that add up to 1.
        probabilities = torch.softmax(logits, dim=1)

        top_probabilities, top_indices = torch.topk(probabilities, k=5, dim=1)

    print("\n========== PREDICTION ==========")
    print(f"Input text: {user_text!r}")
    print("Top 5 predicted next characters:")

    for probability, index in zip(top_probabilities[0], top_indices[0]):
        character = idx_to_char[index.item()]
        display_character = "\\n" if character == "\n" else character
        print(f"Character: {display_character!r:<4} Probability: {probability.item():.4f}")


def interactive_prediction_loop(model, char_to_idx, idx_to_char):
    """
    Let the user type text and get next-character predictions.
    """
    print("\n========== TRY IT YOURSELF ==========")
    print("Enter some text and the model will predict the next character.")
    print("Type 'quit' to stop.\n")

    while True:
        user_text = input("Enter text: ")

        if user_text.lower() == "quit":
            print("Goodbye!")
            break

        if not user_text:
            print("Please enter at least one character.")
            continue

        try:
            predict_next_character(model, user_text, char_to_idx, idx_to_char)
        except ValueError as error:
            print(error)


def main():
    """
    Run the full character prediction pipeline.
    """
    torch.manual_seed(42)

    text = load_text(DATA_FILE)
    vocab, char_to_idx, idx_to_char = build_vocabulary(text)
    encoded_text = encode_text(text, char_to_idx)
    x, y = create_training_samples(encoded_text, MAX_CONTEXT_LENGTH)

    print_learning_outputs(text, vocab, char_to_idx, idx_to_char, encoded_text, x, y)

    model = CharacterPredictor(
        vocab_size=len(vocab),
        max_context_length=MAX_CONTEXT_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
    )

    print("\n========== MODEL ==========")
    print(model)

    # Print shape flow through the model once for learning.
    sample_batch = x[:4]
    sample_embeddings = model.embedding(sample_batch)
    print("\n========== EMBEDDING SHAPES ==========")
    print(f"Sample input batch shape:      {sample_batch.shape}")
    print(f"Sample embedding batch shape:  {sample_embeddings.shape}")
    print(
        "Embedding shape means: "
        "[batch_size, max_context_length, embedding_dimension]"
    )

    train_model(model, x, y)
    interactive_prediction_loop(model, char_to_idx, idx_to_char)


if __name__ == "__main__":
    main()
