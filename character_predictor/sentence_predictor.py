"""
Next-Sentence Predictor from scratch using Python and PyTorch.

Project goal:
Given one sentence, predict the next sentence from a small dataset.

Example:
Input:  "Text can be split into smaller tokens."
Output: "Tokens are converted into integer ids."

This is a beginner-friendly educational model. It is not a chatbot and it does
not generate brand-new sentences. Instead, it learns which sentence from the
training data is likely to come next.
"""

from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.optim as optim


DATA_FILE = Path(__file__).parent / "sentence_data.txt"
MAX_WORDS = 10
EMBEDDING_DIM = 16
HIDDEN_DIM = 64
EPOCHS = 300
LEARNING_RATE = 0.01
PRINT_EVERY = 30


def load_sentences(file_path):
    """
    Load the dataset as a list of sentences.

    Each line in sentence_data.txt is treated as one sentence. Sentence order is
    important because training samples are built from neighboring sentences.
    """
    text = file_path.read_text(encoding="utf-8")
    sentences = [line.strip() for line in text.splitlines() if line.strip()]
    return sentences


def tokenize_sentence(sentence):
    """
    Tokenize a sentence into lowercase word tokens.

    This uses a simple regular expression:
    - words like "learning"
    - numbers like "123"
    - punctuation like "."

    Example:
    "I am learning NLP." -> ["i", "am", "learning", "nlp", "."]
    """
    return re.findall(r"[a-zA-Z0-9]+|[^\w\s]", sentence.lower())


def build_word_vocabulary(sentences):
    """
    Build a vocabulary of unique word/punctuation tokens.

    We add a special <PAD> token at index 0. Padding is used when a sentence has
    fewer than MAX_WORDS tokens, so all examples can fit into one tensor.
    """
    all_tokens = []
    for sentence in sentences:
        all_tokens.extend(tokenize_sentence(sentence))

    vocab = ["<PAD>"] + sorted(set(all_tokens))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    return vocab, word_to_idx, idx_to_word


def encode_sentence(sentence, word_to_idx):
    """
    Convert a sentence into integer token IDs.

    Neural networks cannot directly process strings, so every token must become
    a number first.
    """
    tokens = tokenize_sentence(sentence)
    encoded = [word_to_idx[token] for token in tokens]
    return encoded


def pad_or_truncate(encoded_sentence, max_words):
    """
    Make every encoded sentence the same length.

    If the sentence is too short, add <PAD> IDs to the end.
    If the sentence is too long, keep only the first max_words tokens.
    """
    encoded_sentence = encoded_sentence[:max_words]
    padding_needed = max_words - len(encoded_sentence)
    return encoded_sentence + [0] * padding_needed


def create_training_samples(sentences, word_to_idx, max_words):
    """
    Create input-target pairs using neighboring sentences.

    Example:
    Sentence 0 -> Sentence 1
    Sentence 1 -> Sentence 2
    Sentence 2 -> Sentence 3

    The input is an encoded sentence. The target is a class ID representing the
    next sentence's position in the dataset.
    """
    inputs = []
    targets = []

    for sentence_index in range(len(sentences) - 1):
        current_sentence = sentences[sentence_index]
        next_sentence_index = sentence_index + 1

        encoded = encode_sentence(current_sentence, word_to_idx)
        padded = pad_or_truncate(encoded, max_words)

        inputs.append(padded)
        targets.append(next_sentence_index)

    x = torch.tensor(inputs, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)

    return x, y


class NextSentencePredictor(nn.Module):
    """
    A simple neural network for next-sentence prediction.

    Architecture:
    1. Embedding layer converts token IDs into vectors.
    2. Mean pooling combines token vectors into one sentence vector.
    3. Linear hidden layer learns patterns from the sentence vector.
    4. Output layer predicts one class for each possible sentence.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, number_of_sentences):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, number_of_sentences)

    def forward(self, x):
        """
        Forward pass through the network.

        Input shape:
        [batch_size, max_words]

        Embedding shape:
        [batch_size, max_words, embedding_dim]

        Output shape:
        [batch_size, number_of_sentences]
        """
        embedded = self.embedding(x)

        # Mean pooling creates one vector for the whole sentence by averaging
        # the word embeddings. This is simple and beginner-friendly.
        sentence_vector = embedded.mean(dim=1)

        hidden_output = torch.relu(self.hidden(sentence_vector))
        logits = self.output(hidden_output)
        return logits


def print_learning_outputs(sentences, vocab, word_to_idx, idx_to_word, x, y):
    """
    Print intermediate outputs so the learning pipeline is visible.
    """
    print("\n========== SENTENCES ==========")
    for index, sentence in enumerate(sentences):
        print(f"{index}: {sentence}")

    print("\n========== WORD VOCABULARY ==========")
    print(f"Vocabulary size: {len(vocab)}")
    print(vocab)

    print("\n========== word_to_idx ==========")
    print(word_to_idx)

    print("\n========== idx_to_word ==========")
    print(idx_to_word)

    print("\n========== ENCODED SENTENCE EXAMPLE ==========")
    example_sentence = sentences[0]
    print(f"Sentence: {example_sentence!r}")
    print(f"Tokens:   {tokenize_sentence(example_sentence)}")
    print(f"Encoded:  {encode_sentence(example_sentence, word_to_idx)}")

    print("\n========== TRAINING SAMPLES ==========")
    for i in range(min(8, len(x))):
        input_sentence_id = i
        target_sentence_id = y[i].item()
        print(
            f"Input: {sentences[input_sentence_id]!r}\n"
            f"Target: {sentences[target_sentence_id]!r}\n"
        )

    print("\n========== TENSOR SHAPES ==========")
    print(f"Input tensor x shape:  {x.shape}")
    print(f"Target tensor y shape: {y.shape}")
    print("x shape means: [number_of_samples, max_words]")
    print("y shape means: [number_of_samples]")


def train_model(model, x, y):
    """
    Train the model using CrossEntropyLoss and Adam.
    """
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n========== TRAINING ==========")
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_function(logits, y)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"Epoch {epoch:03d}/{EPOCHS} - Loss: {loss.item():.4f}")


def prepare_user_sentence(user_sentence, word_to_idx):
    """
    Encode user input for prediction.

    Unknown words are skipped. This keeps the demo simple, but in larger
    projects you would usually add an <UNK> token for unknown words.
    """
    tokens = tokenize_sentence(user_sentence)
    known_tokens = [token for token in tokens if token in word_to_idx]

    if not known_tokens:
        raise ValueError("No known tokens found. Try a sentence similar to the dataset.")

    encoded = [word_to_idx[token] for token in known_tokens]
    padded = pad_or_truncate(encoded, MAX_WORDS)
    return torch.tensor([padded], dtype=torch.long), known_tokens


def predict_next_sentence(model, user_sentence, word_to_idx, sentences):
    """
    Predict and print the top 5 next sentences with probabilities.
    """
    model.eval()

    input_tensor, known_tokens = prepare_user_sentence(user_sentence, word_to_idx)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        top_probabilities, top_indices = torch.topk(probabilities, k=5, dim=1)

    print("\n========== PREDICTION ==========")
    print(f"Input sentence: {user_sentence!r}")
    print(f"Known tokens used: {known_tokens}")
    print("Top 5 predicted next sentences:")

    for probability, index in zip(top_probabilities[0], top_indices[0]):
        sentence_id = index.item()
        print(f"{probability.item():.4f} -> {sentences[sentence_id]}")


def interactive_prediction_loop(model, word_to_idx, sentences):
    """
    Let the user type a sentence and get next-sentence predictions.
    """
    print("\n========== TRY IT YOURSELF ==========")
    print("Enter a sentence and the model will predict the next sentence.")
    print("Type 'quit' to stop.\n")

    while True:
        user_sentence = input("Enter sentence: ")

        if user_sentence.lower() == "quit":
            print("Goodbye!")
            break

        if not user_sentence.strip():
            print("Please enter a sentence.")
            continue

        try:
            predict_next_sentence(model, user_sentence, word_to_idx, sentences)
        except ValueError as error:
            print(error)


def main():
    """
    Run the full next-sentence prediction pipeline.
    """
    torch.manual_seed(42)

    sentences = load_sentences(DATA_FILE)
    vocab, word_to_idx, idx_to_word = build_word_vocabulary(sentences)
    x, y = create_training_samples(sentences, word_to_idx, MAX_WORDS)

    print_learning_outputs(sentences, vocab, word_to_idx, idx_to_word, x, y)

    model = NextSentencePredictor(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        number_of_sentences=len(sentences),
    )

    print("\n========== MODEL ==========")
    print(model)

    sample_batch = x[:4]
    sample_embeddings = model.embedding(sample_batch)
    print("\n========== EMBEDDING SHAPES ==========")
    print(f"Sample input batch shape:      {sample_batch.shape}")
    print(f"Sample embedding batch shape:  {sample_embeddings.shape}")

    train_model(model, x, y)
    interactive_prediction_loop(model, word_to_idx, sentences)


if __name__ == "__main__":
    main()
