import { pipeline, env, AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// Configure transformers env to use CDN instead of local models
env.allowLocalModels = false;

// Global variables for model and tokenizer
let tokenizer = null;
let extractor = null;

// Game State
const state = {
    modelLoaded: false,
    activeTab: 'duel-tab',
    duel: {
        targetWord: 'bank',
        targetContext: 'A', // 'A' or 'B'
        vocabData: null,    // loaded polysemy data
        embeddings: {
            contextA: null,
            contextB: null
        }
    },
    navigator: {
        activeMissionIdx: 0,
        startWord: 'Planet',
        targetWord: 'Computer',
        startEmbedding: null,
        targetEmbedding: null,
        history: [], // Array of { word: string, embedding: Float32Array, simToTarget: number, simToStart: number }
        steps: 0
    },
    sandbox: {
        sentence: '',
        tokens: [],
        attentionMatrix: []
    }
};

// Polysemy Word Data
const polysemyData = {
    bank: {
        word: "bank",
        contextA: {
            sentence: "The river bank was covered in soft green moss.",
            meaning: "Edge of a river or body of water"
        },
        contextB: {
            sentence: "I deposited my check at the commercial bank.",
            meaning: "Financial institution"
        }
    },
    apple: {
        word: "apple",
        contextA: {
            sentence: "She baked a fresh red apple into the pie.",
            meaning: "The round fruit of an apple tree"
        },
        contextB: {
            sentence: "He bought a new Apple computer for graphic design.",
            meaning: "The consumer electronics and software company"
        }
    },
    date: {
        word: "date",
        contextA: {
            sentence: "They went on a romantic dinner date on Friday night.",
            meaning: "A social or romantic meeting"
        },
        contextB: {
            sentence: "Please write the current date at the top of the form.",
            meaning: "A specific day of the month and year"
        }
    },
    key: {
        word: "key",
        contextA: {
            sentence: "He inserted the brass key into the door lock.",
            meaning: "A physical metal instrument to open a lock"
        },
        contextB: {
            sentence: "Education is the key to unlocking future success.",
            meaning: "A crucial or vital element / solution"
        }
    },
    cell: {
        word: "cell",
        contextA: {
            sentence: "The human body is made of billions of animal cells.",
            meaning: "The basic structural and functional unit of life"
        },
        contextB: {
            sentence: "The guard locked the prisoner in a dark cell.",
            meaning: "A small room in a prison"
        }
    }
};

// Missions for Navigator
const navigatorMissions = [
    { start: "Planet", target: "Computer" },
    { start: "Forest", target: "City" },
    { start: "Coffee", target: "Sleep" },
    { start: "Ocean", target: "Desert" },
    { start: "Cat", target: "Dog" }
];

/* ==========================================================================
   INITIALIZATION & MODEL LOADER
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initAccordion();
    initEventListeners();
    loadModel();
});

// Setup Tab Navigation
function initTabs() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            state.activeTab = tabId;

            navButtons.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(tabId).classList.add('active');

            // Handle tab-specific redraws
            if (tabId === 'navigator-tab') {
                setTimeout(drawNavigatorMap, 100);
            } else if (tabId === 'explainer-tab') {
                updateSandbox();
            }
        });
    });
}

// Setup Accordion for Explainer
function initAccordion() {
    const headers = document.querySelectorAll('.accordion-header');
    headers.forEach(header => {
        header.addEventListener('click', () => {
            const item = header.parentElement;
            const isActive = item.classList.contains('active');
            
            // Close all items
            document.querySelectorAll('.accordion-item').forEach(i => i.classList.remove('active'));
            
            // Open clicked item if it wasn't active
            if (!isActive) {
                item.classList.add('active');
            }
        });
    });
}

// Terminal Logger
function addTerminalLog(message, type = 'muted') {
    const logBox = document.getElementById('terminal-log');
    if (!logBox) return;

    const line = document.createElement('div');
    line.className = `log-line text-${type}`;
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    line.innerHTML = `<span style="color:#64748b">[${time}]</span> ${message}`;
    
    logBox.appendChild(line);
    logBox.scrollTop = logBox.scrollHeight;
}

// Load Model using Transformers.js
async function loadModel() {
    addTerminalLog('Initializing model pipeline...', 'muted');
    const progressMap = {};

    try {
        // Load tokenizer
        tokenizer = await AutoTokenizer.from_pretrained('Xenova/all-MiniLM-L6-v2');
        addTerminalLog('Tokenizer loaded successfully!', 'success');

        // Load pipeline
        extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
            progress_callback: (data) => {
                const fileName = data.file ? data.file.split('/').pop() : 'model file';
                if (data.status === 'downloading') {
                    addTerminalLog(`Downloading file: ${fileName}`, 'muted');
                } else if (data.status === 'progress') {
                    if (data.file) {
                        progressMap[data.file] = data.progress;
                    }
                    
                    const keys = Object.keys(progressMap);
                    let sum = 0;
                    for (const key of keys) {
                        sum += progressMap[key];
                    }
                    const totalProgress = keys.length > 0 ? Math.round(sum / keys.length) : 0;
                    
                    // Update UI progress
                    document.getElementById('loading-progress-bar').style.width = `${totalProgress}%`;
                    document.getElementById('loading-percentage').innerText = `${totalProgress}%`;
                    document.getElementById('loading-stage').innerText = `DOWNLOADING MODEL WEIGHTS: ${Math.round(totalProgress)}%`;
                } else if (data.status === 'ready') {
                    addTerminalLog(`Loaded file: ${fileName}`, 'success');
                }
            }
        });

        addTerminalLog('BERT Transformer model loaded successfully!', 'success');
        state.modelLoaded = true;

        // Warm up and initialize games
        await initPolysemyGame();
        await initNavigatorGame();
        
        // Hide loading overlay
        setTimeout(() => {
            const overlay = document.getElementById('loading-overlay');
            overlay.style.opacity = 0;
            setTimeout(() => overlay.remove(), 500);
        }, 800);

    } catch (err) {
        addTerminalLog(`ERROR loading model: ${err.message}`, 'danger');
        document.getElementById('loading-stage').innerText = 'LOAD ERROR. PLEASE REFRESH.';
        console.error(err);
    }
}

/* ==========================================================================
   MATH UTILITIES
   ========================================================================== */

function dotProduct(vecA, vecB) {
    let product = 0;
    const len = vecA.length;
    for (let i = 0; i < len; i++) {
        product += vecA[i] * vecB[i];
    }
    return product;
}

function magnitude(vec) {
    let sum = 0;
    const len = vec.length;
    for (let i = 0; i < len; i++) {
        sum += vec[i] * vec[i];
    }
    return Math.sqrt(sum);
}

function cosineSimilarity(vecA, vecB) {
    const magA = magnitude(vecA);
    const magB = magnitude(vecB);
    if (magA === 0 || magB === 0) return 0;
    return dotProduct(vecA, vecB) / (magA * magB);
}

// Simple deterministic hash function for Y-axis coordinates
function stringHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    return hash;
}

// Extracts the token embedding vector from raw output tensor
function getWordEmbedding(tensor, tokenIdx) {
    const shape = tensor.dims; // [1, sequence_length, 384]
    const dim = shape[2];
    const data = tensor.data;
    const offset = tokenIdx * dim;
    return data.slice(offset, offset + dim);
}

// Computes mean pooled sentence embedding
function meanPool(tensor) {
    const shape = tensor.dims;
    const numTokens = shape[1];
    const dim = shape[2];
    const data = tensor.data;
    
    const pooled = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
        let sum = 0;
        for (let t = 0; t < numTokens; t++) {
            sum += data[t * dim + d];
        }
        pooled[d] = sum / numTokens;
    }
    return pooled;
}

/* ==========================================================================
   GAME MODE 1: POLYSEMY DUEL
   ========================================================================== */

async function initPolysemyGame() {
    state.duel.vocabData = polysemyData[state.duel.targetWord];
    updateDuelUIWithWord();
}

function updateDuelUIWithWord() {
    const data = state.duel.vocabData;
    
    // Update word labels
    document.querySelectorAll('.target-word-highlight').forEach(el => el.innerText = data.word);
    document.getElementById('status-target-word').innerText = data.word;
    
    // Set Target context sentences
    document.getElementById('context-a-sentence').innerHTML = highlightWord(data.contextA.sentence, data.word);
    document.getElementById('context-a-meaning').innerText = data.contextA.meaning;

    document.getElementById('context-b-sentence').innerHTML = highlightWord(data.contextB.sentence, data.word);
    document.getElementById('context-b-meaning').innerText = data.contextB.meaning;

    // Reset input
    const textarea = document.getElementById('duel-user-input');
    textarea.value = '';
    textarea.placeholder = `Type a sentence containing the word "${data.word}" (e.g. to match Context ${state.duel.targetContext})...`;
    
    // Reset indicators
    validateDuelInput();
    
    // Hide results & show placeholder
    document.getElementById('duel-results').classList.add('hidden');
    document.getElementById('duel-placeholder').classList.remove('hidden');
}

function highlightWord(sentence, word) {
    const regex = new RegExp(`\\b(${word}s?)\\b`, 'gi');
    return sentence.replace(regex, '<strong>$1</strong>');
}

// Validator for input text
function validateDuelInput() {
    const textarea = document.getElementById('duel-user-input');
    const inputVal = textarea.value.toLowerCase();
    const targetWord = state.duel.targetWord.toLowerCase();
    const indicator = document.getElementById('duel-word-status');
    const submitBtn = document.getElementById('duel-submit-btn');

    // Basic check if word is in sentence
    const regex = new RegExp(`\\b${targetWord}s?\\b`, 'i');
    const isValid = regex.test(inputVal);

    if (isValid) {
        indicator.innerHTML = `<i class="fa-solid fa-circle-check text-green"></i> Sentence contains "${state.duel.targetWord}"`;
        submitBtn.removeAttribute('disabled');
    } else {
        indicator.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-yellow"></i> Sentence must contain "${state.duel.targetWord}"`;
        submitBtn.setAttribute('disabled', 'true');
    }
}

// Run Polysemy Duel Calculation
async function runPolysemyDuel() {
    const submitBtn = document.getElementById('duel-submit-btn');
    submitBtn.setAttribute('disabled', 'true');
    submitBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Extracting Embeddings...`;

    const userSentence = document.getElementById('duel-user-input').value;
    const targetWord = state.duel.targetWord;

    try {
        // 1. Process Context A
        const contextASent = state.duel.vocabData.contextA.sentence;
        const tokensA = await tokenizer(contextASent);
        const idsA = Array.from(tokensA.input_ids.data);
        const strA = idsA.map(id => tokenizer.decode([id]));
        const idxA = findTargetTokenIndex(strA, targetWord);
        const featA = await extractor(contextASent, { pooling: 'none' });
        const vectorA = getWordEmbedding(featA, idxA);

        // 2. Process Context B
        const contextBSent = state.duel.vocabData.contextB.sentence;
        const tokensB = await tokenizer(contextBSent);
        const idsB = Array.from(tokensB.input_ids.data);
        const strB = idsB.map(id => tokenizer.decode([id]));
        const idxB = findTargetTokenIndex(strB, targetWord);
        const featB = await extractor(contextBSent, { pooling: 'none' });
        const vectorB = getWordEmbedding(featB, idxB);

        // 3. Process User Sentence
        const tokensUser = await tokenizer(userSentence);
        const idsUser = Array.from(tokensUser.input_ids.data);
        const strUser = idsUser.map(id => tokenizer.decode([id]));
        const idxUser = findTargetTokenIndex(strUser, targetWord);

        if (idxUser === -1) {
            alert(`Could not isolate "${targetWord}" in your sentence. Make sure it is spelled correctly as a distinct word.`);
            submitBtn.removeAttribute('disabled');
            submitBtn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Compute Vector Similarity`;
            return;
        }

        const featUser = await extractor(userSentence, { pooling: 'none' });
        const vectorUser = getWordEmbedding(featUser, idxUser);

        // 4. Calculate similarities
        const simA = cosineSimilarity(vectorUser, vectorA);
        const simB = cosineSimilarity(vectorUser, vectorB);

        // 5. Display results
        displayDuelResults(simA, simB, strA, idxA, strB, idxB, strUser, idxUser, userSentence);

    } catch (err) {
        alert("An error occurred during embedding calculations: " + err.message);
        console.error(err);
    } finally {
        submitBtn.removeAttribute('disabled');
        submitBtn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Compute Vector Similarity`;
    }
}

function findTargetTokenIndex(tokens, targetWord) {
    const cleanTarget = targetWord.toLowerCase().trim();
    for (let i = 0; i < tokens.length; i++) {
        const cleanTok = tokens[i].toLowerCase().replace('##', '').trim();
        if (cleanTok === cleanTarget) return i;
    }
    // Fallback subword contains
    for (let i = 0; i < tokens.length; i++) {
        const cleanTok = tokens[i].toLowerCase().replace('##', '').trim();
        if (cleanTok.includes(cleanTarget) || cleanTarget.includes(cleanTok)) return i;
    }
    return -1;
}

function displayDuelResults(simA, simB, strA, idxA, strB, idxB, strUser, idxUser, userSentence) {
    document.getElementById('duel-placeholder').classList.add('hidden');
    document.getElementById('duel-results').classList.remove('hidden');

    // 1. Determine target similarities
    const target = state.duel.targetContext;
    const similarityToTarget = target === 'A' ? simA : simB;
    const targetPercent = Math.max(0, Math.min(100, Math.round(similarityToTarget * 100)));

    // 2. Set radial percentage
    const radial = document.getElementById('score-radial');
    radial.style.setProperty('--percent', targetPercent);
    document.getElementById('score-value').innerText = `${targetPercent}%`;

    // 3. Verdict Messages
    const verdictEl = document.getElementById('result-verdict');
    const descEl = document.getElementById('result-explanation');

    const activeMeaning = target === 'A' ? state.duel.vocabData.contextA.meaning : state.duel.vocabData.contextB.meaning;
    const otherMeaning = target === 'B' ? state.duel.vocabData.contextA.meaning : state.duel.vocabData.contextB.meaning;

    if (similarityToTarget > 0.88) {
        verdictEl.innerHTML = `<i class="fa-solid fa-circle-check text-green"></i> Perfect Match!`;
        descEl.innerText = `Outstanding! Your contextual vector points directly to the target meaning ("${activeMeaning}"). BERT successfully aligned your sentence structures.`;
    } else if (similarityToTarget > 0.75) {
        verdictEl.innerHTML = `<i class="fa-solid fa-circle-check text-blue"></i> Great Match!`;
        descEl.innerText = `Well done! Your sentence matched the intended context ("${activeMeaning}"). The embedding has a high similarity score.`;
    } else {
        // Check if closer to the wrong meaning
        const wrongSim = target === 'A' ? simB : simA;
        if (wrongSim > similarityToTarget) {
            verdictEl.innerHTML = `<i class="fa-solid fa-circle-xmark text-purple"></i> Drifted off course!`;
            descEl.innerText = `Your sentence actually matches the other meaning ("${otherMeaning}") closer! Modify your structure to emphasize the river/finance aspect.`;
        } else {
            verdictEl.innerHTML = `<i class="fa-solid fa-circle-question text-yellow"></i> Ambiguous Context`;
            descEl.innerText = `BERT is having trouble categorizing the meaning of "${state.duel.targetWord}". Try making the sentence more descriptive to resolve ambiguity.`;
        }
    }

    // 4. Update breakdown bars
    document.getElementById('sim-score-a').innerText = simA.toFixed(4);
    document.getElementById('sim-bar-a').style.width = `${Math.max(0, simA * 100)}%`;

    document.getElementById('sim-score-b').innerText = simB.toFixed(4);
    document.getElementById('sim-bar-b').style.width = `${Math.max(0, simB * 100)}%`;

    // 5. Render sentence tokens
    renderTokensList('tokens-display-a', strA, idxA, 'blue');
    renderTokensList('tokens-display-b', strB, idxB, 'purple');
    renderTokensList('tokens-display-user', strUser, idxUser, target === 'A' ? 'blue' : 'purple');

    // 6. Draw 2D Vector Projection
    drawEmbeddingProjection(simA, simB);
}

function renderTokensList(containerId, tokens, targetIdx, highlightColor) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    tokens.forEach((tok, i) => {
        const span = document.createElement('span');
        span.className = 'tok';
        
        if (i === 0 || i === tokens.length - 1) {
            span.className += ' special';
        }
        
        if (i === targetIdx) {
            span.className += highlightColor === 'blue' ? ' target-highlight' : ' target-highlight-b';
        }

        span.innerText = tok;
        container.appendChild(span);
    });
}

function drawEmbeddingProjection(simA, simB) {
    const canvas = document.getElementById('duel-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw background graph lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let i = 40; i < canvas.width; i += 40) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, canvas.height);
        ctx.stroke();
    }
    for (let i = 40; i < canvas.height; i += 40) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(canvas.width, i);
        ctx.stroke();
    }

    // Positions of endpoints
    const nodeA = { x: 70, y: 110, label: 'Context A (River)' };
    const nodeB = { x: 330, y: 110, label: 'Context B (Finance)' };
    
    // Set labels based on vocab
    const vocab = state.duel.targetWord;
    if (vocab === 'apple') {
        nodeA.label = 'Fruit';
        nodeB.label = 'Tech Company';
    } else if (vocab === 'date') {
        nodeA.label = 'Social Date';
        nodeB.label = 'Calendar Date';
    } else if (vocab === 'key') {
        nodeA.label = 'Lock Tool';
        nodeB.label = 'Critical Element';
    } else if (vocab === 'cell') {
        nodeA.label = 'Biological Cell';
        nodeB.label = 'Prison Cell';
    }

    // Interpolate Player Node Position
    // simA and simB are cosine similarities (~0.3 to 0.95)
    // Map them to x-axis progress:
    let diff = simB - simA; // range: -0.6 to +0.6
    let scaleDiff = diff * 1.5; // expand contrast
    let w = (scaleDiff + 1) / 2; // scale to 0..1
    w = Math.max(0.15, Math.min(0.85, w)); // Clamp to avoid overriding labels

    const playerX = nodeA.x + w * (nodeB.x - nodeA.x);
    
    // Y position represents overall similarity (how accurate the sentence is)
    // If similarities are low, push Y down.
    const maxSim = Math.max(simA, simB);
    const playerY = 190 - Math.max(0.2, Math.min(1.0, maxSim)) * 120; // range 70 to 190

    // Draw connection line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(nodeA.x, nodeA.y);
    ctx.lineTo(nodeB.x, nodeB.y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw Context A Node
    ctx.shadowBlur = 10;
    ctx.shadowColor = 'rgba(0, 242, 254, 0.6)';
    ctx.fillStyle = '#00f2fe';
    ctx.beginPath();
    ctx.arc(nodeA.x, nodeA.y, 8, 0, Math.PI * 2);
    ctx.fill();

    // Draw Context B Node
    ctx.shadowColor = 'rgba(157, 78, 221, 0.6)';
    ctx.fillStyle = '#9d4edd';
    ctx.beginPath();
    ctx.arc(nodeB.x, nodeB.y, 8, 0, Math.PI * 2);
    ctx.fill();

    // Draw Player Node
    const target = state.duel.targetContext;
    const playerColor = target === 'A' ? '#00f2fe' : '#9d4edd';
    ctx.shadowColor = playerColor;
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = playerColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(playerX, playerY, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Reset shadow
    ctx.shadowBlur = 0;

    // Draw Labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = 'bold 10px Orbitron';
    ctx.textAlign = 'center';
    
    ctx.fillText(nodeA.label.toUpperCase(), nodeA.x, nodeA.y - 18);
    ctx.fillText(nodeB.label.toUpperCase(), nodeB.x, nodeB.y - 18);

    ctx.fillStyle = '#fff';
    ctx.fillText('YOUR EMBEDDING', playerX, playerY - 18);
    
    // Draw dashed lines from player to reference nodes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(playerX, playerY);
    ctx.lineTo(nodeA.x, nodeA.y);
    ctx.moveTo(playerX, playerY);
    ctx.lineTo(nodeB.x, nodeB.y);
    ctx.stroke();
    ctx.setLineDash([]);
}

/* ==========================================================================
   GAME MODE 2: SEMANTIC SPACE NAVIGATOR
   ========================================================================== */

async function initNavigatorGame() {
    const mission = navigatorMissions[state.navigator.activeMissionIdx];
    state.navigator.startWord = mission.start;
    state.navigator.targetWord = mission.target;
    state.navigator.steps = 0;
    state.navigator.history = [];

    document.getElementById('nav-start-word').innerText = mission.start;
    document.getElementById('nav-target-word').innerText = mission.target;
    document.getElementById('nav-step-count').innerText = '0';
    document.getElementById('nav-current-sim').innerText = '0%';
    
    const historyList = document.getElementById('nav-history-list');
    historyList.innerHTML = `<div class="history-empty">No nodes traversed yet. Type a word and press Enter to launch.</div>`;

    document.getElementById('nav-input').value = '';

    // Calculate Endpoints Embeddings
    const featStart = await extractor(mission.start);
    state.navigator.startEmbedding = meanPool(featStart);

    const featTarget = await extractor(mission.target);
    state.navigator.targetEmbedding = meanPool(featTarget);

    drawNavigatorMap();
}

async function handleNavigatorSubmit() {
    const inputField = document.getElementById('nav-input');
    const submitBtn = document.getElementById('nav-submit-btn');
    const word = inputField.value.trim();

    if (!word) return;

    // Disable input during extraction
    inputField.disabled = true;
    submitBtn.disabled = true;

    try {
        // Extract pooled sentence embedding
        const feat = await extractor(word);
        const embedding = meanPool(feat);

        // Compute similarities
        const simToStart = cosineSimilarity(embedding, state.navigator.startEmbedding);
        const simToTarget = cosineSimilarity(embedding, state.navigator.targetEmbedding);

        // Increment steps
        state.navigator.steps++;
        document.getElementById('nav-step-count').innerText = state.navigator.steps;

        const currentSimPercent = Math.max(0, Math.min(100, Math.round(simToTarget * 100)));
        document.getElementById('nav-current-sim').innerText = `${currentSimPercent}%`;

        // Add to history
        const node = { word, embedding, simToTarget, simToStart };
        state.navigator.history.push(node);

        // Update list UI
        updateNavigatorHistoryList();
        
        // Redraw Canvas
        drawNavigatorMap();

        // Check Victory Condition
        if (simToTarget >= 0.85) {
            setTimeout(() => {
                alert(`🎉 MISSION COMPLETED! You connected "${state.navigator.startWord}" to "${state.navigator.targetWord}" in ${state.navigator.steps} steps! Current similarity: ${(simToTarget * 100).toFixed(1)}%`);
                // Move to next mission
                state.navigator.activeMissionIdx = (state.navigator.activeMissionIdx + 1) % navigatorMissions.length;
                initNavigatorGame();
            }, 300);
        }

        // Reset inputs
        inputField.value = '';

    } catch (err) {
        alert("Error generating embedding: " + err.message);
    } finally {
        inputField.disabled = false;
        submitBtn.disabled = false;
        inputField.focus();
    }
}

function updateNavigatorHistoryList() {
    const list = document.getElementById('nav-history-list');
    list.innerHTML = '';

    state.navigator.history.forEach((node, i) => {
        const item = document.createElement('div');
        item.className = 'history-node';
        item.innerHTML = `
            <span class="node-idx">NODE #${i+1}</span>
            <span class="node-text">${node.word}</span>
            <span class="node-sim">${(node.simToTarget * 100).toFixed(1)}% Sim</span>
        `;
        list.appendChild(item);
    });
    list.scrollTop = list.scrollHeight;
}

function drawNavigatorMap() {
    const canvas = document.getElementById('nav-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Make canvas responsive to its bounds
    const rect = canvas.parentNode.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Draw dark starfield background
    ctx.fillStyle = '#04070e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw space dust particles (stars)
    ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
    const startSeed = stringHash(state.navigator.startWord);
    for (let i = 0; i < 40; i++) {
        const starX = Math.abs(stringHash(state.navigator.startWord + i) % canvas.width);
        const starY = Math.abs(stringHash(state.navigator.targetWord + i) % canvas.height);
        const size = (i % 3 === 0) ? 2 : 1;
        ctx.fillRect(starX, starY, size, size);
    }

    // Horizontal centerline
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height/2);
    ctx.lineTo(canvas.width, canvas.height/2);
    ctx.stroke();

    // Endpoint positions
    const startNode = { x: 70, y: canvas.height/2 };
    const targetNode = { x: canvas.width - 70, y: canvas.height/2 };

    // Draw connecting axis
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 10]);
    ctx.beginPath();
    ctx.moveTo(startNode.x, startNode.y);
    ctx.lineTo(targetNode.x, targetNode.y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Plot History Steps
    let prevPoint = startNode;
    state.navigator.history.forEach((node, i) => {
        // Calculate coordinates
        let w = (node.simToTarget - node.simToStart + 1) / 2;
        w = Math.max(0.12, Math.min(0.88, w)); // Clamp to keep on screen
        const px = startNode.x + w * (targetNode.x - startNode.x);

        // Deterministic offset Y based on string hash. Points closer to endpoints have smaller offsets
        const hashVal = stringHash(node.word);
        const maxOffset = (1 - Math.max(node.simToStart, node.simToTarget)) * 140;
        const py = (canvas.height/2) + Math.sin(hashVal) * maxOffset;

        const currentPoint = { x: px, y: py };

        // Draw path connecting lines
        ctx.strokeStyle = 'rgba(57, 255, 20, 0.3)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(prevPoint.x, prevPoint.y);
        ctx.lineTo(currentPoint.x, currentPoint.y);
        ctx.stroke();

        // Draw particle node
        ctx.shadowBlur = 8;
        ctx.shadowColor = 'rgba(57, 255, 20, 0.6)';
        ctx.fillStyle = '#39ff14';
        ctx.beginPath();
        ctx.arc(currentPoint.x, currentPoint.y, 5, 0, Math.PI * 2);
        ctx.fill();

        // Draw label for the node
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px Space Grotesk';
        ctx.textAlign = 'center';
        ctx.fillText(node.word, currentPoint.x, currentPoint.y - 10);

        prevPoint = currentPoint;
    });

    // Draw line from last node to target (with faded dashed line)
    if (state.navigator.history.length > 0) {
        ctx.strokeStyle = 'rgba(157, 78, 221, 0.2)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 4]);
        ctx.beginPath();
        ctx.moveTo(prevPoint.x, prevPoint.y);
        ctx.lineTo(targetNode.x, targetNode.y);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // DRAW START NODE
    ctx.shadowBlur = 12;
    ctx.shadowColor = 'rgba(0, 242, 254, 0.6)';
    ctx.fillStyle = '#00f2fe';
    ctx.beginPath();
    ctx.arc(startNode.x, startNode.y, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 11px Orbitron';
    ctx.textAlign = 'center';
    ctx.fillText(state.navigator.startWord.toUpperCase(), startNode.x, startNode.y - 15);

    // DRAW TARGET NODE
    ctx.shadowColor = 'rgba(157, 78, 221, 0.6)';
    ctx.fillStyle = '#9d4edd';
    ctx.beginPath();
    ctx.arc(targetNode.x, targetNode.y, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 11px Orbitron';
    ctx.fillText(state.navigator.targetWord.toUpperCase(), targetNode.x, targetNode.y - 15);

    ctx.shadowBlur = 0;
}

/* ==========================================================================
   INTERACTIVE BERT SIMULATOR ("HOW BERT WORKS" TAB)
   ========================================================================== */

async function updateSandbox() {
    const sentence = document.getElementById('sandbox-input').value;
    if (!sentence || !tokenizer) return;

    state.sandbox.sentence = sentence;

    // Tokenize
    const result = await tokenizer(sentence);
    const ids = Array.from(result.input_ids.data);
    const tokens = ids.map(id => tokenizer.decode([id]));
    state.sandbox.tokens = tokens;

    // Update Tokenizer Output Visualizer
    const container = document.getElementById('sandbox-tokens');
    container.innerHTML = '';
    tokens.forEach((tok, i) => {
        const span = document.createElement('span');
        span.className = 'sandbox-tok';
        if (tok.startsWith('##') || tok === '[CLS]' || tok === '[SEP]') {
            span.className += ' special';
        }
        span.innerText = tok;
        span.setAttribute('data-idx', i);
        container.appendChild(span);
    });

    // Update Attention column lists
    const leftCol = document.getElementById('attention-words-left');
    const rightCol = document.getElementById('attention-words-right');
    leftCol.innerHTML = '';
    rightCol.innerHTML = '';

    tokens.forEach((tok, i) => {
        const itemLeft = document.createElement('div');
        itemLeft.className = 'attn-word-node';
        itemLeft.innerText = tok;
        itemLeft.setAttribute('data-idx', i);
        leftCol.appendChild(itemLeft);

        const itemRight = document.createElement('div');
        itemRight.className = 'attn-word-node';
        itemRight.innerText = tok;
        itemRight.setAttribute('data-idx', i);
        rightCol.appendChild(itemRight);
    });

    // Generate simulated attention matrix
    generateSimulatedAttentionMatrix(tokens);

    // Initial draw (select first content token or CLS)
    const selectIdx = tokens.length > 1 ? 1 : 0;
    selectAttentionLeftNode(selectIdx);
}

// Generates a mock but grammatically sensible attention matrix
function generateSimulatedAttentionMatrix(tokens) {
    const len = tokens.length;
    const matrix = [];

    // Heuristics lists
    const pronouns = ["it", "he", "she", "they", "this", "that", "its", "their", "his", "her", "who", "whom"];
    const verbs = ["is", "are", "was", "were", "rely", "relies", "use", "uses", "attention", "mechanism", "mechanisms", "embed", "embedding", "embeddings", "train", "training", "works"];
    const nouns = ["transformer", "architecture", "mechanism", "mechanisms", "attention", "vector", "vectors", "word", "words", "sentence", "sentences"];

    for (let i = 0; i < len; i++) {
        const row = new Float32Array(len);
        let rowSum = 0;

        const currentTok = tokens[i].toLowerCase().replace('##', '').trim();

        for (let j = 0; j < len; j++) {
            const comparisonTok = tokens[j].toLowerCase().replace('##', '').trim();

            let weight = 0.05; // background attention

            if (i === j) {
                weight += 0.35; // self-attending
            }
            if (j === 0) {
                weight += 0.15; // CLS context attending
            }
            if (Math.abs(i - j) === 1) {
                weight += 0.15; // adjacent context
            }

            // Heuristic links
            // Pronoun references Nouns
            if (pronouns.includes(currentTok) && nouns.includes(comparisonTok)) {
                weight += 0.4;
            }
            // Verb links to Subject/Object (nouns)
            if (verbs.includes(currentTok) && nouns.includes(comparisonTok)) {
                weight += 0.35;
            }
            // Nouns link to Adjectives (nearby words with suffixes or common words)
            if (nouns.includes(currentTok) && (comparisonTok.endsWith('al') || comparisonTok.endsWith('tive') || comparisonTok.endsWith('ive') || comparisonTok.endsWith('lar'))) {
                weight += 0.3;
            }

            row[j] = weight;
            rowSum += weight;
        }

        // Normalize row sum to 1.0
        for (let j = 0; j < len; j++) {
            row[j] /= rowSum;
        }
        matrix.push(row);
    }
    state.sandbox.attentionMatrix = matrix;
}

function selectAttentionLeftNode(idx) {
    // Highlight left node
    const leftNodes = document.querySelectorAll('#attention-words-left .attn-word-node');
    leftNodes.forEach(node => {
        const nodeIdx = parseInt(node.getAttribute('data-idx'));
        if (nodeIdx === idx) {
            node.classList.add('selected-left');
        } else {
            node.classList.remove('selected-left');
        }
    });

    // Highlight sandbox tab tokens as well
    const sandboxToks = document.querySelectorAll('.sandbox-tok');
    sandboxToks.forEach(node => {
        const nodeIdx = parseInt(node.getAttribute('data-idx'));
        if (nodeIdx === idx) {
            node.classList.add('active-attn');
        } else {
            node.classList.remove('active-attn');
        }
    });

    // Update weights in right column
    const rightNodes = document.querySelectorAll('#attention-words-right .attn-word-node');
    const weights = state.sandbox.attentionMatrix[idx] || [];

    rightNodes.forEach(node => {
        const nodeIdx = parseInt(node.getAttribute('data-idx'));
        const w = weights[nodeIdx] || 0.05;

        // Set weight visual opacity and color variable
        node.style.setProperty('--attn-val', w * 1.5);
        node.classList.add('highlight-right');
    });

    // Draw connecting lines on canvas
    drawAttentionLines(idx, weights);
}

function drawAttentionLines(leftIdx, weights) {
    const canvas = document.getElementById('attention-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Make canvas sharp on retina displays
    const rect = canvas.parentNode.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const leftNodes = document.querySelectorAll('#attention-words-left .attn-word-node');
    const rightNodes = document.querySelectorAll('#attention-words-right .attn-word-node');

    if (leftNodes.length === 0 || rightNodes.length === 0 || !leftNodes[leftIdx]) return;

    // Get position of the active left node (Y center relative to canvas)
    const leftNodeRect = leftNodes[leftIdx].getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();
    
    const startPoint = {
        x: 0,
        y: (leftNodeRect.top + leftNodeRect.height/2) - canvasRect.top
    };

    rightNodes.forEach(node => {
        const nodeIdx = parseInt(node.getAttribute('data-idx'));
        const w = weights[nodeIdx] || 0;

        if (w < 0.04) return; // skip drawing faint lines

        const rightNodeRect = node.getBoundingClientRect();
        const endPoint = {
            x: canvas.width,
            y: (rightNodeRect.top + rightNodeRect.height/2) - canvasRect.top
        };

        // Draw cubic bezier curve
        ctx.strokeStyle = `rgba(0, 242, 254, ${w * 1.8})`;
        ctx.lineWidth = Math.max(0.5, w * 8);
        ctx.beginPath();
        ctx.moveTo(startPoint.x, startPoint.y);
        ctx.bezierCurveTo(
            canvas.width * 0.4, startPoint.y,
            canvas.width * 0.6, endPoint.y,
            endPoint.x, endPoint.y
        );
        ctx.stroke();
    });
}

/* ==========================================================================
   EVENT LISTENERS
   ========================================================================== */

function initEventListeners() {
    // 1. Polysemy Dropdown Selector
    const wordSelect = document.getElementById('word-select');
    wordSelect.addEventListener('change', (e) => {
        state.duel.targetWord = e.target.value;
        state.duel.vocabData = polysemyData[state.duel.targetWord];
        updateDuelUIWithWord();
    });

    // 2. Select Target Context A / B buttons
    const btnA = document.getElementById('select-target-a');
    const btnB = document.getElementById('select-target-b');
    const contextCardA = document.getElementById('context-a-card');
    const contextCardB = document.getElementById('context-b-card');

    btnA.addEventListener('click', () => {
        state.duel.targetContext = 'A';
        btnA.classList.add('active-a');
        btnB.classList.remove('active-b');
        
        contextCardA.classList.add('selected-a');
        contextCardB.classList.remove('selected-b');
        
        const textarea = document.getElementById('duel-user-input');
        textarea.placeholder = `Type a sentence containing the word "${state.duel.targetWord}" to match Context A...`;
    });

    btnB.addEventListener('click', () => {
        state.duel.targetContext = 'B';
        btnB.classList.add('active-b');
        btnA.classList.remove('active-a');

        contextCardB.classList.add('selected-b');
        contextCardA.classList.remove('selected-a');

        const textarea = document.getElementById('duel-user-input');
        textarea.placeholder = `Type a sentence containing the word "${state.duel.targetWord}" to match Context B...`;
    });

    // 3. Polysemy Textarea change listener
    const duelTextarea = document.getElementById('duel-user-input');
    duelTextarea.addEventListener('input', validateDuelInput);

    // 4. Polysemy Submit Button
    const duelSubmit = document.getElementById('duel-submit-btn');
    duelSubmit.addEventListener('click', runPolysemyDuel);

    // 5. Space Navigator word input listener
    const navInput = document.getElementById('nav-input');
    navInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleNavigatorSubmit();
        }
    });

    const navSubmit = document.getElementById('nav-submit-btn');
    navSubmit.addEventListener('click', handleNavigatorSubmit);

    // 6. Mission Shuffle Button
    const shuffleBtn = document.getElementById('new-mission-btn');
    shuffleBtn.addEventListener('click', () => {
        state.navigator.activeMissionIdx = (state.navigator.activeMissionIdx + 1) % navigatorMissions.length;
        initNavigatorGame();
    });

    // 7. Sandbox Explainer input listener
    const sandboxInput = document.getElementById('sandbox-input');
    sandboxInput.addEventListener('input', updateSandbox);

    // 8. Sandbox Token Hover trigger for Attention map
    const sandboxTokensContainer = document.getElementById('sandbox-tokens');
    sandboxTokensContainer.addEventListener('mouseover', (e) => {
        if (e.target.classList.contains('sandbox-tok')) {
            const idx = parseInt(e.target.getAttribute('data-idx'));
            selectAttentionLeftNode(idx);
        }
    });

    const leftColContainer = document.getElementById('attention-words-left');
    leftColContainer.addEventListener('mouseover', (e) => {
        if (e.target.classList.contains('attn-word-node')) {
            const idx = parseInt(e.target.getAttribute('data-idx'));
            selectAttentionLeftNode(idx);
        }
    });

    // 9. Resize handler to redraw maps
    window.addEventListener('resize', () => {
        if (state.activeTab === 'navigator-tab') {
            drawNavigatorMap();
        } else if (state.activeTab === 'duel-tab' && !document.getElementById('duel-results').classList.contains('hidden')) {
            // Get current similarities from display
            const simA = parseFloat(document.getElementById('sim-score-a').innerText);
            const simB = parseFloat(document.getElementById('sim-score-b').innerText);
            drawEmbeddingProjection(simA, simB);
        } else if (state.activeTab === 'explainer-tab') {
            const selectedLeft = document.querySelector('#attention-words-left .attn-word-node.selected-left');
            if (selectedLeft) {
                const idx = parseInt(selectedLeft.getAttribute('data-idx'));
                const weights = state.sandbox.attentionMatrix[idx] || [];
                drawAttentionLines(idx, weights);
            }
        }
    });
}
