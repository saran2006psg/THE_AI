"""Encoding logic for the educational BPE tokenizer."""


class BPEEncoder:
    """Converts text into BPE tokens and token IDs using learned merge rules."""

    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = [tuple(pair) for pair in merges]

    def _apply_single_merge(self, tokens, pair_to_merge):
        """Apply one learned merge rule from left to right."""
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

        return new_tokens

    def encode(self, text):
        """Encode text by replaying merge rules in their learned order."""
        tokens = list(text)

        for pair in self.merges:
            tokens = self._apply_single_merge(tokens, pair)

        token_ids = []
        for token in tokens:
            if token not in self.vocab:
                raise ValueError(
                    f"Token {token!r} is not in the vocabulary. "
                    "Train with a corpus that contains this character first."
                )
            token_ids.append(self.vocab[token])

        return tokens, token_ids
