import json
import os
import numpy as np
from src.dataset import corpus, large_corpus, preprocess, build_vocab, generate_training_pairs
from src.model import Word2VecSkipGram
from src.train import train_neg_sampling, build_unigram_table

def export_data():
    print("==================================================")
    print("       EXPORTING EMBEDDINGS FOR WEB EXPLORER")
    print("==================================================")
    
    # 1. Clean sentences
    raw_sentences = [s.strip() for s in large_corpus.split('.') if len(s.strip()) > 10]
    
    # Preprocess full corpus to build vocab
    words = preprocess(large_corpus)
    max_vocab_size = 250
    word_to_index, index_to_word, vocab_size = build_vocab(words, max_vocab_size)
    pairs = generate_training_pairs(words, word_to_index, window_size=2)
    
    # 2. Train Skip-Gram with Negative Sampling
    embedding_dim = 16
    epochs = 150
    model = Word2VecSkipGram(vocab_size, embedding_dim)
    unigram_table = build_unigram_table(words, word_to_index)
    
    print(f"Training Word2Vec SGNS...")
    print(f"-> Vocabulary Size: {vocab_size}")
    print(f"-> Embedding Dim: {embedding_dim}")
    print(f"-> Training Epochs: {epochs}")
    train_neg_sampling(model, pairs, unigram_table, num_negatives=5, epochs=epochs, lr=0.05)
    
    # 3. Filter sentences that are clean and fit BERT-style [MASK] task
    game_sentences = []
    for sentence in raw_sentences:
        s_words = preprocess(sentence)
        if 5 <= len(s_words) <= 18:
            # Keep sentence if at least 75% of its words exist in our top vocab
            in_vocab_words = [w for w in s_words if w in word_to_index]
            if len(in_vocab_words) >= len(s_words) * 0.75:
                clean_s = " ".join(s_words)
                game_sentences.append(clean_s)
                
    game_sentences = sorted(list(set(game_sentences))) # Deduplicate and sort
    
    # 4. JSON Payload
    embeddings_list = model.W_in.tolist()
    
    export_payload = {
        "vocab": sorted(list(word_to_index.keys())),
        "word_to_index": word_to_index,
        "embeddings": embeddings_list,
        "sentences": game_sentences
    }
    
    os.makedirs("explorer_game", exist_ok=True)
    export_path = os.path.join("explorer_game", "embeddings.json")
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
        
    print(f"\n[SUCCESS] Exported vocab, weights, and {len(game_sentences)} sentences to {export_path}!")

if __name__ == "__main__":
    export_data()
