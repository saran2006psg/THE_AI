"""Training logic for a beginner-friendly Byte Pair Encoding tokenizer.

BPE training starts with very small tokens, usually individual characters.
Then it repeatedly finds the most common adjacent pair of tokens and merges
that pair into a new larger token. The order of merges is important because
encoding later replays these exact rules in the same sequence.
"""

from collections import Counter
import json


class BPETrainer:
    """Learns BPE merge rules and builds a vocabulary from a text corpus."""

    def __init__(self, vocab_size=100, verbose=True):
        self.vocab_size = vocab_size
        self.verbose = verbose
        self.vocab = {}
        self.id_to_token = {}
        self.merges = []
        self.merge_history = []

    def _text_to_tokens(self, text):
        """Represent the entire corpus as a flat list of character tokens."""
        return list(text)

    def _build_initial_vocab(self, tokens):
        """Create token IDs for every unique starting character."""
        unique_tokens = sorted(set(tokens))
        self.vocab = {token: token_id for token_id, token in enumerate(unique_tokens)}
        self.id_to_token = {token_id: token for token, token_id in self.vocab.items()}

    def count_pairs(self, tokens):
        """Count every adjacent token pair in the current token sequence."""
        pair_counts = Counter()

        for index in range(len(tokens) - 1):
            pair = (tokens[index], tokens[index + 1])
            pair_counts[pair] += 1

        return pair_counts

    def merge_pair(self, tokens, pair_to_merge):
        """Replace every non-overlapping occurrence of a pair with one token."""
        merged_token = "".join(pair_to_merge)
        new_tokens = []
        index = 0

        while index < len(tokens):
            can_merge = (
                index < len(tokens) - 1
                and tokens[index] == pair_to_merge[0]
                and tokens[index + 1] == pair_to_merge[1]
            )

            if can_merge:
                new_tokens.append(merged_token)
                index += 2
            else:
                new_tokens.append(tokens[index])
                index += 1

        return new_tokens, merged_token

    def _print_pair_frequencies(self, pair_counts):
        """Print pair frequencies from most common to least common."""
        print("\nPair frequencies:")
        for pair, frequency in pair_counts.most_common():
            print(f"  {pair!r}: {frequency}")

    def _print_final_vocabulary(self):
        """Display the complete vocabulary sorted by token ID."""
        print("\nFinal vocabulary:")
        for token_id in sorted(self.id_to_token):
            print(f"  {token_id}: {self.id_to_token[token_id]!r}")

    def train_from_text(self, text):
        """Train BPE on a string and return learned vocabulary and merges."""
        tokens = self._text_to_tokens(text)
        self._build_initial_vocab(tokens)
        self.merges = []
        self.merge_history = []

        if self.verbose:
            print("Starting BPE training")
            print(f"Initial character-level token count: {len(tokens)}")
            print(f"Initial vocabulary size: {len(self.vocab)}")

        # Stop when the desired vocabulary size is reached or no pairs remain.
        while len(self.vocab) < self.vocab_size:
            pair_counts = self.count_pairs(tokens)

            if not pair_counts:
                if self.verbose:
                    print("\nNo more adjacent pairs to merge.")
                break

            best_pair, best_frequency = pair_counts.most_common(1)[0]

            # A frequency of 1 means the pair is not repeated. We can still
            # merge it to reach the requested vocabulary size, but showing this
            # condition helps learners understand when useful compression ends.
            new_tokens, merged_token = self.merge_pair(tokens, best_pair)

            # If the merged token already exists, adding it would not grow the
            # vocabulary. This rare case can happen because tokens are strings.
            if merged_token in self.vocab:
                if self.verbose:
                    print(f"\nSkipping {best_pair!r}; token {merged_token!r} already exists.")
                break

            new_token_id = len(self.vocab)
            self.vocab[merged_token] = new_token_id
            self.id_to_token[new_token_id] = merged_token
            self.merges.append(list(best_pair))

            history_entry = {
                "step": len(self.merges),
                "pair": list(best_pair),
                "merged_token": merged_token,
                "frequency": best_frequency,
                "vocab_size": len(self.vocab),
                "token_count": len(new_tokens),
            }
            self.merge_history.append(history_entry)

            if self.verbose:
                print("\n" + "=" * 60)
                print(f"Merge step {history_entry['step']}")
                self._print_pair_frequencies(pair_counts)
                print(f"\nMost frequent pair: {best_pair!r} ({best_frequency} times)")
                print(f"Created token: {merged_token!r}")
                print(f"Vocabulary size after merge: {len(self.vocab)}")
                print(f"Corpus token count after merge: {len(new_tokens)}")

            tokens = new_tokens

        if self.verbose:
            self._print_final_vocabulary()

        return self.vocab, self.merges, self.merge_history

    def train_from_file(self, corpus_path):
        """Read a corpus text file and train on its contents."""
        with open(corpus_path, "r", encoding="utf-8") as corpus_file:
            text = corpus_file.read()

        return self.train_from_text(text)

    def save(self, vocab_path, merges_path):
        """Save vocabulary, reverse vocabulary, merges, and history as JSON."""
        vocab_data = {
            "vocab": self.vocab,
            "id_to_token": {str(token_id): token for token_id, token in self.id_to_token.items()},
            "merge_history": self.merge_history,
        }

        with open(vocab_path, "w", encoding="utf-8") as vocab_file:
            json.dump(vocab_data, vocab_file, indent=2, ensure_ascii=False)

        with open(merges_path, "w", encoding="utf-8") as merges_file:
            json.dump(self.merges, merges_file, indent=2, ensure_ascii=False)
