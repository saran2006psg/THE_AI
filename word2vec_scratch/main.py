import numpy as np
from src.dataset import corpus, large_corpus, preprocess, build_vocab, generate_training_pairs
from src.model import Word2VecSkipGram
from src.train import train_softmax, train_neg_sampling, build_unigram_table, evaluate_similarities
from utils.visualize import plot_embeddings
import os

def main():
    print("==================================================")
    print("     WORD2VEC IMPLEMENTATION FROM SCRATCH")
    print("==================================================")
    
    # Toggle this to False if you want to run on the tiny corpus instead
    USE_LARGE_DATASET = True
    selected_corpus = large_corpus if USE_LARGE_DATASET else corpus
    
    # 1. Preprocess & Vocabulary mapping
    print("\n--- Phase 1: Preprocessing & Data Prep ---")
    words = preprocess(selected_corpus)
    
    if len(words) == 0:
        print("\n[ERROR] The corpus is empty!")
        print("-> If you want to use the Wikipedia dataset, make sure you have saved the file 'src/data.txt' in your editor.")
        print("-> If you want to use the tiny sample dataset instead, edit 'main.py' and set 'USE_LARGE_DATASET = False'.")
        return
        
    # Configure settings based on dataset size
    if USE_LARGE_DATASET:
        max_vocab_size = 300
        epochs = 200
        test_pairs = [
            ("language", "model"),
            ("neural", "network"),
            ("deep", "learning"),
            ("transformer", "attention"),
            ("word", "embeddings")
        ]
    else:
        max_vocab_size = 20
        epochs = 800
        test_pairs = [
            ("king", "queen"), 
            ("king", "strong"), 
            ("queen", "wise"), 
            ("strong", "wise"),
            ("king", "wise")
        ]
        
    word_to_index, index_to_word, vocab_size = build_vocab(words, max_vocab_size)
    pairs = generate_training_pairs(words, word_to_index, window_size=2)
    
    print(f"Selected corpus length (words): {len(words)}")
    print(f"Vocab size: {vocab_size}")
    print(f"Num training pairs: {len(pairs)}")
    
    # 2. Skip-Gram with Softmax
    print("\n--- Phase 2 & 3: Skip-Gram with Softmax ---")
    embedding_dim = 8
    model_softmax = Word2VecSkipGram(vocab_size, embedding_dim)
    
    print("\nInitial similarities:")
    evaluate_similarities(model_softmax, word_to_index, test_pairs)
    
    # Train
    train_softmax(model_softmax, pairs, epochs=epochs, lr=0.05)
    
    print("\nAfter training similarities (Softmax):")
    evaluate_similarities(model_softmax, word_to_index, test_pairs)
    
    # Plot Softmax embeddings
    plot_embeddings(model_softmax.W_in, word_to_index, "embeddings_softmax.png")
    
    # 3. Skip-Gram with Negative Sampling
    print("\n--- Phase 4: Skip-Gram with Negative Sampling ---")
    model_ns = Word2VecSkipGram(vocab_size, embedding_dim)
    
    print("\nInitial similarities:")
    evaluate_similarities(model_ns, word_to_index, test_pairs)
    
    # Build Unigram table
    unigram_table = build_unigram_table(words, word_to_index)
    
    # Train
    train_neg_sampling(model_ns, pairs, unigram_table, num_negatives=5, epochs=epochs, lr=0.05)
    
    print("\nAfter training similarities (Negative Sampling):")
    evaluate_similarities(model_ns, word_to_index, test_pairs)
    
    # Plot NS embeddings
    plot_embeddings(model_ns.W_in, word_to_index, "embeddings_neg_sampling.png")

if __name__ == "__main__":
    main()
