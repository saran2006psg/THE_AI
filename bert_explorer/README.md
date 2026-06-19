# BERT Semantic Odyssey: The Contextual Embedding Game

Welcome to the **BERT Semantic Odyssey**, an interactive, visually stunning educational game designed to demonstrate how **Bidirectional Encoder Representations from Transformers (BERT)** generate contextual embeddings. 

This project runs **entirely inside your web browser**. It leverages [Transformers.js](https://huggingface.co/docs/transformers.js/index) to load and execute a lightweight, 384-dimensional BERT-based embedding model (`all-MiniLM-L6-v2`) via WebAssembly, requiring zero installation of Python or deep learning frameworks on your host machine.

---

## 🚀 How to Run Locally

Because the application uses modern JavaScript ES Modules and loads WASM binaries via CDN, web browsers block local execution directly from files (using `file://`) due to CORS security policies. You must serve the directory using a web server.

### Option 1: Python HTTP Server (Recommended & Easiest)
If you have Python installed, run this command in your terminal:

```bash
cd bert_explorer
python -m http.server 8000
```

Then open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

### Option 2: Node.js (npx)
If you have Node.js installed, run:

```bash
cd bert_explorer
npx serve
```

---

## 🎮 Game Modes

### 1. Polysemy Duel
*   **The Concept**: Showcases how BERT produces *different* embedding vectors for the exact same word depending on its surrounding context.
*   **The Mission**: Choose a target word (like `bank`, `apple`, `date`, `key`, or `cell`) and select which context (A or B) you want to match. Craft a custom sentence of your own.
*   **How it Works**: 
    *   BERT tokenizes your sentence and extracts the hidden state embedding for the target word.
    *   It calculates the Cosine Similarity between your word vector and the pre-computed contexts.
    *   It plots your vector in a projected 2D semantic space, displaying how it drifts toward the meaning you described.

### 2. Semantic Space Navigator
*   **The Concept**: Explores semantic sentence similarity.
*   **The Mission**: Navigate from a **Start Concept** (e.g., `Planet`) to a **Destination Concept** (e.g., `Computer`) by typing intermediate words or phrases.
*   **How it Works**: 
    *   For each word/phrase you type, BERT performs mean pooling over all tokens to get a single sentence embedding.
    *   It compares your vector with the Start and Destination vectors.
    *   It plots your step-by-step progress on a 2D Canvas space, drawing a starfield path towards the destination. Reach $\ge 85\%$ similarity to complete the mission!

---

## 🧠 Under the Hood: "How BERT Works" Panel
Includes a visual dashboard detailing:
1.  **Static vs. Contextual Embeddings**: Why Word2Vec struggles with dual-meanings, and how BERT's bidirectional transformer solves this.
2.  **WordPiece Tokenizer Sandbox**: See your sentences split into WordPieces (e.g., `embedding` $\rightarrow$ `em`, `##bed`, `##dings`) with special tokens like `[CLS]` (Classification) and `[SEP]` (Separator).
3.  **Self-Attention Simulator**: An interactive connection map. Hover over tokens to see how they look at surrounding words (using grammatical attention heuristics) to capture contextual meanings.

---

## 🛠️ Technology Stack
*   **Frontend**: Semantic HTML5, Vanilla JavaScript (ES modules)
*   **Styling**: Custom Vanilla CSS (Glassmorphism layout, radial glow backgrounds, glowing hover indicators)
*   **Inference Engine**: `@xenova/transformers` (Transformers.js v2) loading ONNX weights of `all-MiniLM-L6-v2` (~23MB model download)
*   **Visualizations**: Custom HTML5 Canvas (glowing nodes, paths, starfields, bezier attention links)
