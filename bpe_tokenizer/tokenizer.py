"""High-level BPE tokenizer wrapper.

This file connects the trainer, encoder, and decoder so users can work with
one simple class from the CLI or from another Python program.
"""

import json

from decoder import BPEDecoder
from encoder import BPEEncoder
from trainer import BPETrainer


class BPETokenizer:
    """Beginner-friendly Byte Pair Encoding tokenizer."""

    def __init__(self, vocab_size=100, verbose=True):
        self.vocab_size = vocab_size
        self.verbose = verbose
        self.vocab = {}
        self.id_to_token = {}
        self.merges = []
        self.merge_history = []

    def train(self, corpus_path, vocab_path="vocab.json", merges_path="merges.json"):
        """Train on a corpus file and save vocabulary plus merge rules."""
        trainer = BPETrainer(vocab_size=self.vocab_size, verbose=self.verbose)
        self.vocab, self.merges, self.merge_history = trainer.train_from_file(corpus_path)
        self.id_to_token = trainer.id_to_token
        trainer.save(vocab_path, merges_path)

    def load(self, vocab_path="vocab.json", merges_path="merges.json"):
        """Load a previously trained tokenizer from JSON files."""
        with open(vocab_path, "r", encoding="utf-8") as vocab_file:
            vocab_data = json.load(vocab_file)

        with open(merges_path, "r", encoding="utf-8") as merges_file:
            self.merges = json.load(merges_file)

        self.vocab = vocab_data["vocab"]
        self.id_to_token = {int(token_id): token for token_id, token in vocab_data["id_to_token"].items()}
        self.merge_history = vocab_data.get("merge_history", [])

    def encode(self, text):
        """Return both BPE tokens and token IDs for input text."""
        self._require_trained()
        encoder = BPEEncoder(self.vocab, self.merges)
        return encoder.encode(text)

    def decode(self, token_ids):
        """Return original text from token IDs."""
        self._require_trained()
        decoder = BPEDecoder(self.id_to_token)
        return decoder.decode(token_ids)

    def display_vocabulary(self):
        """Print all tokens sorted by token ID."""
        self._require_trained()
        print("\nVocabulary:")
        for token_id in sorted(self.id_to_token):
            print(f"  {token_id}: {self.id_to_token[token_id]!r}")

    def display_merge_history(self):
        """Print every learned merge rule in order."""
        self._require_trained()
        print("\nMerge history:")
        if not self.merge_history:
            print("  No merges were learned.")
            return

        for entry in self.merge_history:
            pair = tuple(entry["pair"])
            print(
                f"  Step {entry['step']}: {pair!r} -> {entry['merged_token']!r} "
                f"(frequency={entry['frequency']}, vocab_size={entry['vocab_size']})"
            )

    def show_statistics(self, text):
        """Compare character-level size with BPE tokenized size."""
        tokens, token_ids = self.encode(text)
        character_count = len(text)
        bpe_count = len(tokens)
        compression_ratio = bpe_count / character_count if character_count else 0

        print("\nToken statistics:")
        print(f"  Character-level token count: {character_count}")
        print(f"  BPE token count: {bpe_count}")
        print(f"  Compression ratio (BPE / characters): {compression_ratio:.3f}")
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")

    def _require_trained(self):
        """Raise a helpful error if encode/decode is used before training/loading."""
        if not self.vocab or not self.id_to_token:
            raise RuntimeError("Tokenizer is not trained or loaded yet.")
