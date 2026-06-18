"""Interactive command-line interface for the BPE tokenizer."""

import os

from tokenizer import BPETokenizer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(BASE_DIR, "corpus.txt")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.json")
MERGES_PATH = os.path.join(BASE_DIR, "merges.json")
DEMO_TEXT = "The transformer architecture is unbelievably powerful"


def read_token_ids():
    """Read token IDs from the user as a comma-separated or space-separated list."""
    raw_ids = input("Enter token IDs (comma or space separated): ").strip()
    pieces = raw_ids.replace(",", " ").split()
    return [int(piece) for piece in pieces]


def load_if_available(tokenizer):
    """Load saved tokenizer files if they already exist."""
    if os.path.exists(VOCAB_PATH) and os.path.exists(MERGES_PATH):
        tokenizer.load(VOCAB_PATH, MERGES_PATH)
        return True
    return False


def train_tokenizer():
    """Ask for vocabulary size, train, save files, and run the required demo."""
    requested_size = input("Vocabulary size (press Enter for 80): ").strip()
    vocab_size = int(requested_size) if requested_size else 80

    tokenizer = BPETokenizer(vocab_size=vocab_size, verbose=True)
    tokenizer.train(CORPUS_PATH, VOCAB_PATH, MERGES_PATH)

    print("\nSaved vocabulary and merges:")
    print(f"  {VOCAB_PATH}")
    print(f"  {MERGES_PATH}")

    print(f"\nDemonstration encode: {DEMO_TEXT!r}")
    tokenizer.show_statistics(DEMO_TEXT)
    return tokenizer


def print_menu():
    """Display CLI options."""
    print("\nBPE Tokenizer CLI")
    print("1. Train tokenizer on corpus.txt")
    print("2. Encode text")
    print("3. Decode token IDs")
    print("4. Display vocabulary")
    print("5. Display merge history")
    print("6. Exit")


def main():
    """Run the interactive CLI loop."""
    tokenizer = BPETokenizer(verbose=True)
    load_if_available(tokenizer)

    while True:
        print_menu()
        choice = input("Choose an option: ").strip()

        try:
            if choice == "1":
                tokenizer = train_tokenizer()
            elif choice == "2":
                if not tokenizer.vocab:
                    print("No saved tokenizer found. Please train first.")
                    continue
                text = input("Enter text to encode: ")
                tokenizer.show_statistics(text)
            elif choice == "3":
                if not tokenizer.vocab:
                    print("No saved tokenizer found. Please train first.")
                    continue
                token_ids = read_token_ids()
                print(f"Decoded text: {tokenizer.decode(token_ids)!r}")
            elif choice == "4":
                tokenizer.display_vocabulary()
            elif choice == "5":
                tokenizer.display_merge_history()
            elif choice == "6":
                print("Goodbye!")
                break
            else:
                print("Please choose a number from 1 to 6.")
        except (RuntimeError, ValueError, FileNotFoundError) as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
