"""Decoding logic for the educational BPE tokenizer."""


class BPEDecoder:
    """Converts token IDs back into text."""

    def __init__(self, id_to_token):
        self.id_to_token = {int(token_id): token for token_id, token in id_to_token.items()}

    def decode(self, token_ids):
        """Decode by mapping each ID to its token and concatenating tokens."""
        tokens = []

        for token_id in token_ids:
            if token_id not in self.id_to_token:
                raise ValueError(f"Token ID {token_id} is not in the vocabulary.")
            tokens.append(self.id_to_token[token_id])

        return "".join(tokens)
