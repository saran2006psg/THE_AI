import numpy as np
import random
import os
import re

# Step 2: Tiny Sample Dataset Selection
corpus = "the king is a strong man. the queen is a wise woman."

def generate_large_corpus(num_sentences=1000):
    """
    Generates a large synthetic corpus of related sentences.
    """
    subjects = ["king", "queen", "prince", "princess", "man", "woman", "boy", "girl", "dog", "cat", "soldier", "knight", "teacher", "doctor"]
    adjectives = ["strong", "wise", "young", "beautiful", "brave", "loyal", "cute", "smart", "kind", "noble", "honest", "gentle"]
    nouns = ["man", "woman", "boy", "girl", "animal", "person", "leader", "warrior", "protector", "friend"]
    verbs = ["is", "seems", "becomes"]
    
    sentences = []
    random.seed(42)  # For reproducibility
    
    for _ in range(num_sentences):
        sub = random.choice(subjects)
        adj = random.choice(adjectives)
        n = random.choice(nouns)
        v = random.choice(verbs)
        
        struct = random.randint(1, 4)
        if struct == 1:
            s = f"the {sub} {v} a {adj} {n}"
        elif struct == 2:
            s = f"the {adj} {sub} {v} a {n}"
        elif struct == 3:
            s = f"the {sub} likes the {adj} {n}"
        else:
            s = f"the {adj} {sub} rules the kingdom"
            
        sentences.append(s)
        
    return ". ".join(sentences) + "."

# Try to load from data.txt if present, otherwise fallback to synthetic
data_file_path = os.path.join(os.path.dirname(__file__), "data.txt")
if os.path.exists(data_file_path):
    with open(data_file_path, "r", encoding="utf-8") as f:
        large_corpus = f.read()
else:
    large_corpus = generate_large_corpus(1000)

def preprocess(text):
    """
    Lowercases the text, removes punctuation, and splits into words.
    """
    text = text.lower()
    # Keep alphanumeric characters and whitespace
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    return words

def build_vocab(words, max_vocab_size=150):
    """
    Builds vocabulary and index mappings from a list of words, keeping the top most frequent words.
    """
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
        
    # Sort by frequency descending
    sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # Keep top words
    top_words = [word for word, count in sorted_words[:max_vocab_size]]
    unique_words = sorted(top_words)
    vocab_size = len(unique_words)
    
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    index_to_word = {i: word for i, word in enumerate(unique_words)}
    
    return word_to_index, index_to_word, vocab_size

def generate_training_pairs(words, word_to_index, window_size=2):
    """
    Generates target-context index pairs based on a sliding window, skipping out-of-vocab words.
    """
    pairs = []
    n = len(words)
    
    for i in range(n):
        if words[i] not in word_to_index:
            continue
        target_idx = word_to_index[words[i]]
        
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        
        for j in range(start, end):
            if i != j:
                if words[j] not in word_to_index:
                    continue
                context_idx = word_to_index[words[j]]
                pairs.append((target_idx, context_idx))
                
    return pairs

def get_one_hot(index, vocab_size):
    """
    Creates a one-hot encoded vector for a given word index.
    """
    one_hot = np.zeros(vocab_size)
    one_hot[index] = 1.0
    return one_hot

if __name__ == "__main__":
    words = preprocess(corpus)
    print("Preprocessed words:", words)
    
    word_to_index, index_to_word, vocab_size = build_vocab(words)
    print(f"\nVocabulary Size: {vocab_size}")
    print(f"word_to_index mapping: {word_to_index}")
    
    pairs = generate_training_pairs(words, word_to_index, window_size=2)
    print(f"\nGenerated {len(pairs)} training pairs.")
    print(f"First 5 pairs (target_idx, context_idx): {pairs[:5]}")
    
    sample_word = "king"
    if sample_word in word_to_index:
        sample_idx = word_to_index[sample_word]
        sample_one_hot = get_one_hot(sample_idx, vocab_size)
        print(f"\nOne-hot vector for '{sample_word}' (index {sample_idx}):\n{sample_one_hot}")
