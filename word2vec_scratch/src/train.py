import numpy as np
from src.dataset import preprocess, build_vocab, generate_training_pairs, get_one_hot, corpus
from src.model import Word2VecSkipGram, cross_entropy_loss
import os

def cosine_similarity(v1, v2):
    """
    Computes cosine similarity between two vectors.
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def train_softmax(model, pairs, epochs, lr=0.05):
    """
    Step 13: Train using standard Softmax
    """
    print("Starting training with Softmax...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        # Shuffle training pairs for Stochastic Gradient Descent
        np.random.shuffle(pairs)
        for target_idx, context_idx in pairs:
            # Forward pass
            y_pred = model.forward(target_idx)
            # Loss calculation
            loss = cross_entropy_loss(y_pred, context_idx)
            epoch_loss += loss
            # Backward pass & weight update
            model.backward_softmax(context_idx, lr)
            
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Avg Loss: {epoch_loss / len(pairs):.4f}")

def build_unigram_table(words, word_to_index, table_size=10000):
    """
    Step 15: Build the Unigram Table for Negative Sampling
    """
    # 1. Count frequencies of words
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
        
    vocab_size = len(word_to_index)
    freqs = np.zeros(vocab_size)
    for word, count in counts.items():
        if word in word_to_index:
            freqs[word_to_index[word]] = count
            
    # 2. Raise frequencies to the power of 3/4 (0.75)
    power_freqs = np.power(freqs, 0.75)
    sum_power_freqs = np.sum(power_freqs)
    probs = power_freqs / sum_power_freqs
    
    # 3. Construct the table array
    table = []
    for idx, prob in enumerate(probs):
        count = int(round(prob * table_size))
        table.extend([idx] * count)
        
    table = np.array(table)
    np.random.shuffle(table)
    return table

def train_neg_sampling(model, pairs, unigram_table, num_negatives, epochs, lr=0.05):
    """
    Step 16: Train using Negative Sampling
    """
    print("\nStarting training with Negative Sampling...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        np.random.shuffle(pairs)
        for target_idx, context_idx in pairs:
            # Sample negative indices from unigram table
            negatives = []
            while len(negatives) < num_negatives:
                neg_idx = np.random.choice(unigram_table)
                # Keep negatives different from target and context
                if neg_idx != target_idx and neg_idx != context_idx:
                    negatives.append(neg_idx)
            
            # Forward pass
            loss = model.forward_negative_sampling(target_idx, context_idx, negatives)
            epoch_loss += loss
            
            # Backward pass & weight update
            model.backward_negative_sampling(lr)
            
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Avg Loss: {epoch_loss / len(pairs):.4f}")

def evaluate_similarities(model, word_to_index, word_pairs):
    """
    Evaluates and prints cosine similarities between word pairs.
    Here we compute target embeddings (W_in). In standard Word2Vec, 
    the final representation is often W_in, or W_in + W_out.T. We will use W_in.
    """
    for w1, w2 in word_pairs:
        if w1 in word_to_index and w2 in word_to_index:
            idx1 = word_to_index[w1]
            idx2 = word_to_index[w2]
            v1 = model.W_in[idx1]
            v2 = model.W_in[idx2]
            sim = cosine_similarity(v1, v2)
            print(f"  Similarity between '{w1}' and '{w2}': {sim:.4f}")
        else:
            print(f"  Word pair '{w1}'-'{w2}' not found in vocabulary.")
