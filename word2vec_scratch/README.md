# Word2Vec from Scratch using NumPy

This project is a modular, step-by-step implementation of the Word2Vec model from scratch using only Python and NumPy. It supports both standard **Skip-Gram with Softmax** and **Skip-Gram with Negative Sampling (SGNS)**.

It is currently configured to read the Wikipedia Large Language Model text dataset from `src/data.txt` to train embeddings and visualize them.

---

## Directory Structure

```text
word2vec_scratch/
├── src/
│   ├── __init__.py
│   ├── dataset.py       # Preprocessing, vocab building, sliding window pair generator
│   ├── model.py         # Word2Vec model class, Softmax and Negative Sampling passes
│   ├── train.py         # SGD training loops, Unigram noise table, cosine similarity
│   └── data.txt         # Wikipedia page raw text (used as the corpus)
├── utils/
│   └── visualize.py     # Manual PCA projection to 2D & matplotlib plotter
├── main.py              # Main execution script
└── README.md            # Execution instructions
```

---

## Requirements

Ensure you have the following installed in your local Python environment:
```bash
pip install numpy matplotlib
```

---

## How to Run

1. Open your terminal or Command Prompt.
2. Navigate to the project directory:
   ```bash
   cd d:\STUDY\PLACMENT\RAG\word2vec_scratch
   ```
3. Run the pipeline:
   ```bash
   python main.py
   ```

---

## Toggling Datasets

By default, the script trains on your Wikipedia text dataset (`src/data.txt`) for **20 epochs** using a vocabulary limit of the top **150** most frequent words.

If you want to train on the tiny synthetic corpus instead (`"the king is a strong man. the queen is a wise woman."`):
1. Open `main.py` in your editor.
2. Locate the variable `USE_LARGE_DATASET` inside the `main()` function:
   ```python
   # Toggle this to False if you want to run on the tiny corpus instead
   USE_LARGE_DATASET = False
   ```
3. Save and run `python main.py` again.

---

## Expected Outputs

Upon completion, the script will:
1. Print the average training loss at each interval.
2. Evaluate and print how the cosine similarity between word pairs (e.g. `("language", "model")` or `("king", "queen")`) changed before and after training.
3. Manually perform PCA on the final embedding matrices and save 2D projection scatter plots:
   - `embeddings_softmax.png` (using Softmax Skip-Gram)
   - `embeddings_neg_sampling.png` (using Negative Sampling SGNS)
