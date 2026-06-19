// Fallback Dataset in case embeddings.json is not found (offline/sandboxed execution)
const fallbackData = {
    vocab: ["language", "model", "transformer", "network", "neural", "deep", "learning", "attention", "data", "training", "word", "embeddings", "artificial", "intelligence", "science", "gpt", "bert", "weights", "parameters", "context"],
    word_to_index: {
        "language": 0, "model": 1, "transformer": 2, "network": 3, "neural": 4,
        "deep": 5, "learning": 6, "attention": 7, "data": 8, "training": 9,
        "word": 10, "embeddings": 11, "artificial": 12, "intelligence": 13, "science": 14,
        "gpt": 15, "bert": 16, "weights": 17, "parameters": 18, "context": 19
    },
    embeddings: [
        [0.15, -0.08, 0.38, 0.85, -0.28, 0.12, 0.48, 0.22], // language
        [0.08, -0.18, 0.32, 0.81, -0.22, 0.18, 0.41, 0.28], // model
        [0.78, 0.12, -0.18, 0.28, 0.48, 0.85, -0.12, 0.38], // transformer
        [-0.28, 0.85, 0.38, -0.08, 0.18, 0.28, 0.78, -0.18], // network
        [-0.38, 0.78, 0.28, -0.18, 0.12, 0.18, 0.68, -0.08], // neural
        [0.18, 0.48, -0.08, 0.38, 0.85, 0.18, 0.28, 0.12], // deep
        [0.28, 0.38, -0.18, 0.48, 0.78, 0.28, 0.38, 0.18], // learning
        [0.85, 0.18, -0.28, 0.12, 0.38, 0.78, -0.18, 0.28], // attention
        [0.38, 0.28, 0.85, 0.18, -0.08, 0.38, 0.48, 0.58], // data
        [0.48, 0.18, 0.78, 0.28, -0.18, 0.48, 0.38, 0.68], // training
        [0.08, -0.28, 0.18, 0.68, -0.38, 0.08, 0.28, 0.18], // word
        [0.18, -0.18, 0.28, 0.78, -0.28, 0.18, 0.38, 0.08], // embeddings
        [-0.48, 0.38, 0.78, 0.08, 0.18, -0.28, 0.48, 0.85], // artificial
        [-0.58, 0.28, 0.68, 0.18, 0.08, -0.38, 0.58, 0.78], // intelligence
        [0.08, 0.58, 0.48, 0.28, 0.18, 0.38, 0.85, -0.08], // science
        [0.68, -0.12, 0.22, 0.78, 0.15, 0.72, 0.35, 0.42], // gpt
        [0.72, -0.15, 0.28, 0.75, 0.12, 0.68, 0.42, 0.38], // bert
        [0.28, 0.18, 0.58, 0.42, -0.15, 0.32, 0.55, 0.62], // weights
        [0.32, 0.22, 0.62, 0.38, -0.12, 0.28, 0.52, 0.58], // parameters
        [0.45, -0.05, 0.32, 0.68, -0.18, 0.25, 0.45, 0.35]  // context
    ],
    sentences: [
        "attention is all you need for transformer models",
        "large language models are trained on text datasets",
        "deep learning uses neural networks to process information",
        "word embeddings represent language semantics in vector spaces",
        "artificial intelligence is changing computer science",
        "transformers weights are updated during training parameters",
        "bert is a masked language model from google research"
    ]
};

// Application State
let modelData = fallbackData;
let semantleSecret = "";
let semantleGuesses = [];
let bertSentence = "";
let bertMaskedWord = "";
let bertFullWords = [];

// DOM Elements
const elVocabSize = document.getElementById("stat-vocab-size");
const elVectorDim = document.getElementById("stat-vector-dim");
const elSourceType = document.getElementById("stat-source-type");
const elLoadStatus = document.getElementById("load-status");

// Initialization
async function init() {
    setupTabNavigation();
    
    try {
        console.log("Attempting to load custom embeddings.json...");
        const response = await fetch("embeddings.json");
        if (!response.ok) throw new Error("File not found");
        modelData = await response.json();
        
        elLoadStatus.textContent = "Custom Wikipedia Data";
        elLoadStatus.className = "status-badge online";
        elSourceType.textContent = "Wiki Upload";
    } catch (err) {
        console.warn("Could not load embeddings.json, falling back to offline pre-baked dataset.", err);
        modelData = fallbackData;
        elLoadStatus.textContent = "Offline (Fallback Data)";
        elLoadStatus.className = "status-badge offline";
        elSourceType.textContent = "Synthetic";
    }
    
    // Set Dashboard Stats
    elVocabSize.textContent = modelData.vocab.length;
    elVectorDim.textContent = modelData.embeddings[0].length;
    
    // Initialize Games
    startSemantleGame();
    setupSemantleForm();
    
    startBertGame();
    setupBertForm();
    
    setupAnalogyForm();
}

// Math Utility: Cosine Similarity
function cosineSimilarity(v1, v2) {
    let dot = 0;
    let norm1 = 0;
    let norm2 = 0;
    for (let i = 0; i < v1.length; i++) {
        dot += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    if (norm1 === 0 || norm2 === 0) return 0;
    return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

// Tab Switching Logic
function setupTabNavigation() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");
    
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));
            
            btn.classList.add("active");
            const tabId = btn.getAttribute("data-tab");
            document.getElementById(`content-${tabId}`).classList.add("active");
        });
    });
}

// ==========================================================================
// GAME MODE 1: SEMANTLE
// ==========================================================================
function startSemantleGame() {
    // Choose a secret word (ensure length > 3 and not a common stop word if possible)
    const candidates = modelData.vocab.filter(w => w.length > 3);
    const randomIndex = Math.floor(Math.random() * candidates.length);
    semantleSecret = candidates[randomIndex];
    semantleGuesses = [];
    
    // Reset GUI
    document.getElementById("semantle-history").innerHTML = `
        <tr class="empty-placeholder"><td colspan="4">No guesses yet. Start playing!</td></tr>
    `;
    document.getElementById("closeness-card").style.display = "none";
    document.getElementById("semantle-msg").textContent = "";
    document.getElementById("input-semantle-guess").value = "";
    
    console.log("Secret word selected:", semantleSecret);
}

function handleSemantleGuess(guessWord) {
    guessWord = guessWord.trim().toLowerCase();
    const msgBox = document.getElementById("semantle-msg");
    
    // Check if word is in vocabulary
    if (!modelData.word_to_index.hasOwnProperty(guessWord)) {
        msgBox.textContent = `"${guessWord}" is not in the vocabulary! Try another word.`;
        msgBox.className = "feedback-msg error";
        return;
    }
    
    // Check if already guessed
    if (semantleGuesses.some(g => g.word === guessWord)) {
        msgBox.textContent = `You already guessed "${guessWord}"!`;
        msgBox.className = "feedback-msg error";
        return;
    }
    
    msgBox.textContent = "";
    
    // Calculate Cosine Similarity
    const secretVec = modelData.embeddings[modelData.word_to_index[semantleSecret]];
    const guessVec = modelData.embeddings[modelData.word_to_index[guessWord]];
    const similarity = cosineSimilarity(secretVec, guessVec);
    
    // Add to guesses list
    semantleGuesses.push({ word: guessWord, similarity: similarity });
    // Sort descending by similarity
    semantleGuesses.sort((a, b) => b.similarity - a.similarity);
    
    // Update Guesses list UI
    renderSemantleHistory();
    
    // Update current similarity bar
    updateSimilarityBar(guessWord, similarity);
    
    // Win Condition
    if (guessWord === semantleSecret) {
        msgBox.textContent = `🎉 AMAZING! You guessed the secret word "${semantleSecret}"!`;
        msgBox.className = "feedback-msg success";
    }
}

function updateSimilarityBar(word, similarity) {
    document.getElementById("closeness-card").style.display = "block";
    const simText = document.getElementById("last-sim-val");
    const simBar = document.getElementById("last-sim-bar");
    const simFeedback = document.getElementById("last-sim-feedback");
    
    simText.textContent = similarity.toFixed(4);
    
    // Convert -1..1 similarity to 0..100% progress width
    const percentage = Math.max(0, Math.min(100, (similarity + 1) * 50));
    simBar.style.width = `${percentage}%`;
    
    if (similarity > 0.95) {
        simFeedback.textContent = "🔥 Found it!";
        simFeedback.style.color = "var(--accent-emerald)";
    } else if (similarity > 0.6) {
        simFeedback.style.color = "var(--accent-red)";
        simFeedback.textContent = "🥵 Extremely Hot!";
    } else if (similarity > 0.3) {
        simFeedback.style.color = "hsl(30, 100%, 55%)";
        simFeedback.textContent = "☀️ Warm!";
    } else if (similarity > 0.1) {
        simFeedback.style.color = "hsl(200, 100%, 65%)";
        simFeedback.textContent = "⛅ Cold";
    } else {
        simFeedback.style.color = "var(--text-muted)";
        simFeedback.textContent = "❄️ Freezing Cold...";
    }
}

function renderSemantleHistory() {
    const tbody = document.getElementById("semantle-history");
    tbody.innerHTML = "";
    
    semantleGuesses.forEach((guess, idx) => {
        const isCorrect = guess.word === semantleSecret;
        const tr = document.createElement("tr");
        tr.className = `guess-row ${isCorrect ? 'correct' : ''}`;
        
        let badgeClass = "cold";
        if (guess.similarity > 0.6) badgeClass = "hot";
        else if (guess.similarity > 0.2) badgeClass = "warm";
        
        tr.innerHTML = `
            <td>${idx + 1}</td>
            <td><strong>${guess.word}</strong></td>
            <td><span class="similarity-badge ${badgeClass}">${guess.similarity.toFixed(4)}</span></td>
            <td>${Math.round((guess.similarity + 1) * 50)}% closeness</td>
        `;
        tbody.appendChild(tr);
    });
}

function setupSemantleForm() {
    const form = document.getElementById("form-semantle");
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        const input = document.getElementById("input-semantle-guess");
        handleSemantleGuess(input.value);
        input.value = "";
    });
    
    document.getElementById("btn-restart-semantle").addEventListener("click", startSemantleGame);
    
    document.getElementById("btn-hint-semantle").addEventListener("click", () => {
        // Find a word from vocabulary with similarity > 0.4 that hasn't been guessed yet
        const secretVec = modelData.embeddings[modelData.word_to_index[semantleSecret]];
        let bestHint = "";
        let bestSim = 0;
        
        modelData.vocab.forEach(word => {
            if (word === semantleSecret || semantleGuesses.some(g => g.word === word)) return;
            const vec = modelData.embeddings[modelData.word_to_index[word]];
            const sim = cosineSimilarity(secretVec, vec);
            
            if (sim > bestSim && sim < 0.9) {
                bestSim = sim;
                bestHint = word;
            }
        });
        
        const msgBox = document.getElementById("semantle-msg");
        if (bestHint) {
            msgBox.textContent = `💡 Hint: Try guessing a word similar to "${bestHint}" (similarity ~ ${bestSim.toFixed(2)})`;
            msgBox.className = "feedback-msg success";
        } else {
            msgBox.textContent = `💡 Hint: Focus on words related to natural language processing!`;
            msgBox.className = "feedback-msg success";
        }
    });
}

// ==========================================================================
// GAME MODE 2: BERT MASKED PREDICTOR
// ==========================================================================
function startBertGame() {
    if (modelData.sentences.length === 0) {
        document.getElementById("bert-sentence").innerHTML = "No sentences found in this vocabulary.";
        return;
    }
    
    // Choose random sentence
    const randIdx = Math.floor(Math.random() * modelData.sentences.length);
    bertSentence = modelData.sentences[randIdx];
    bertFullWords = bertSentence.split(" ");
    
    // Choose a word to mask (prefer words inside our vocabulary, length > 3)
    const validMaskIndices = [];
    bertFullWords.forEach((word, idx) => {
        if (word.length > 3 && modelData.word_to_index.hasOwnProperty(word)) {
            validMaskIndices.push(idx);
        }
    });
    
    if (validMaskIndices.length === 0) {
        // Fallback to random word
        const fallbackIdx = Math.floor(Math.random() * bertFullWords.length);
        bertMaskedWord = bertFullWords[fallbackIdx];
        bertFullWords[fallbackIdx] = "[MASK]";
    } else {
        const maskIdx = validMaskIndices[Math.floor(Math.random() * validMaskIndices.length)];
        bertMaskedWord = bertFullWords[maskIdx];
        bertFullWords[maskIdx] = "[MASK]";
    }
    
    // Render masked sentence
    const displaySentence = bertFullWords.map(w => w === "[MASK]" ? "<code>[MASK]</code>" : w).join(" ");
    document.getElementById("bert-sentence").innerHTML = displaySentence;
    
    // Reset UI
    document.getElementById("bert-msg").textContent = "";
    document.getElementById("input-bert-guess").value = "";
    document.getElementById("bert-predictions").innerHTML = `
        <div class="empty-placeholder">Submit a guess to see MLM context predictions.</div>
    `;
}

function handleBertGuess(guessWord) {
    guessWord = guessWord.trim().toLowerCase();
    const msgBox = document.getElementById("bert-msg");
    
    if (guessWord === bertMaskedWord) {
        msgBox.textContent = "🎉 CORRECT! You filled the mask perfectly!";
        msgBox.className = "feedback-msg success";
    } else {
        msgBox.textContent = `❌ Incorrect. Your guess has been compared. The correct word was "${bertMaskedWord}".`;
        msgBox.className = "feedback-msg error";
    }
    
    // Render Model predictions based on context vectors
    computeBertContextPredictions();
}

function computeBertContextPredictions() {
    // In our Skip-Gram model, the probability of a word in a slot is proportional 
    // to its cosine similarity to the context words surrounding the [MASK].
    // Let's identify the context words (all words in the sentence excluding the masked one)
    const contextWords = bertFullWords.filter(w => w !== "[MASK]" && modelData.word_to_index.hasOwnProperty(w));
    
    if (contextWords.length === 0) {
        document.getElementById("bert-predictions").innerHTML = `
            <div class="empty-placeholder">No context words found in vocabulary to compute scores.</div>
        `;
        return;
    }
    
    // Calculate a score for each word in the vocabulary: average cosine similarity to all context words
    const candidates = [];
    
    modelData.vocab.forEach(word => {
        let totalSim = 0;
        const wordVec = modelData.embeddings[modelData.word_to_index[word]];
        
        contextWords.forEach(ctx => {
            const ctxVec = modelData.embeddings[modelData.word_to_index[ctx]];
            totalSim += cosineSimilarity(wordVec, ctxVec);
        });
        
        const score = totalSim / contextWords.length;
        candidates.push({ word: word, score: score });
    });
    
    // Sort candidates descending by context score
    candidates.sort((a, b) => b.score - a.score);
    
    // Render top 5 predictions
    const container = document.getElementById("bert-predictions");
    container.innerHTML = "";
    
    const top5 = candidates.slice(0, 5);
    top5.forEach(cand => {
        const isTrueWord = cand.word === bertMaskedWord;
        const row = document.createElement("div");
        row.className = "pred-row";
        if (isTrueWord) {
            row.style.border = "1px solid var(--accent-emerald)";
            row.style.background = "rgba(16, 185, 129, 0.05)";
        }
        
        // Convert -1..1 score to 0..100% width
        const widthPercent = Math.max(0, Math.min(100, (cand.score + 1) * 50));
        
        row.innerHTML = `
            <span class="pred-word">${cand.word} ${isTrueWord ? '✅' : ''}</span>
            <div class="pred-bar-container">
                <div class="pred-bar">
                    <div class="pred-fill" style="width: ${widthPercent}%; background: ${isTrueWord ? 'var(--accent-emerald)' : 'var(--accent-purple)'}"></div>
                </div>
                <span class="pred-score">${cand.score.toFixed(2)}</span>
            </div>
        `;
        container.appendChild(row);
    });
}

function setupBertForm() {
    const form = document.getElementById("form-bert");
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        const input = document.getElementById("input-bert-guess");
        handleBertGuess(input.value);
    });
    
    document.getElementById("btn-next-bert").addEventListener("click", startBertGame);
    document.getElementById("btn-reveal-bert").addEventListener("click", () => {
        const msgBox = document.getElementById("bert-msg");
        msgBox.textContent = `The secret masked word was: "${bertMaskedWord}"`;
        msgBox.className = "feedback-msg success";
        computeBertContextPredictions();
    });
}

// ==========================================================================
// TAB 3: VECTOR ANALOGY SOLVER
// ==========================================================================
function solveAnalogy() {
    const wordA = document.getElementById("analogy-a").value.trim().toLowerCase();
    const wordB = document.getElementById("analogy-b").value.trim().toLowerCase();
    const wordC = document.getElementById("analogy-c").value.trim().toLowerCase();
    const resultVal = document.getElementById("analogy-result");
    const container = document.getElementById("analogy-candidates");
    
    // Check vocabulary existence
    const missing = [];
    if (!modelData.word_to_index.hasOwnProperty(wordA)) missing.push(wordA);
    if (!modelData.word_to_index.hasOwnProperty(wordB)) missing.push(wordB);
    if (!modelData.word_to_index.hasOwnProperty(wordC)) missing.push(wordC);
    
    if (missing.length > 0) {
        resultVal.textContent = "???";
        container.innerHTML = `
            <div class="empty-placeholder" style="color: var(--accent-red)">
                Error: The following words are not in the vocabulary: ${missing.join(", ")}
            </div>
        `;
        return;
    }
    
    const vecA = modelData.embeddings[modelData.word_to_index[wordA]];
    const vecB = modelData.embeddings[modelData.word_to_index[wordB]];
    const vecC = modelData.embeddings[modelData.word_to_index[wordC]];
    
    // Calculate target vector: D = A - B + C
    const targetVec = [];
    for (let i = 0; i < vecA.length; i++) {
        targetVec.push(vecA[i] - vecB[i] + vecC[i]);
    }
    
    // Find closest words to target vector in the entire vocabulary
    const candidates = [];
    modelData.vocab.forEach(word => {
        // Skip words that are part of the input query
        if (word === wordA || word === wordB || word === wordC) return;
        
        const vec = modelData.embeddings[modelData.word_to_index[word]];
        const sim = cosineSimilarity(targetVec, vec);
        candidates.push({ word: word, similarity: sim });
    });
    
    // Sort candidates descending by similarity
    candidates.sort((a, b) => b.similarity - a.similarity);
    
    // Display result
    if (candidates.length > 0) {
        const topMatch = candidates[0];
        resultVal.textContent = topMatch.word;
        
        // Render candidates
        container.innerHTML = "";
        candidates.slice(0, 8).forEach(cand => {
            const card = document.createElement("div");
            card.className = "candidate-card";
            card.innerHTML = `
                <span class="candidate-word">${cand.word}</span>
                <span class="candidate-score">${cand.similarity.toFixed(4)}</span>
            `;
            container.appendChild(card);
        });
    } else {
        resultVal.textContent = "None";
        container.innerHTML = `<div class="empty-placeholder">No candidates found.</div>`;
    }
}

function setupAnalogyForm() {
    document.getElementById("btn-solve-analogy").addEventListener("click", solveAnalogy);
}

// Start everything
window.addEventListener("DOMContentLoaded", init);
