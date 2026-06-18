# Bigram Name Generator

A beginner-friendly NLP project that trains a character-level Bigram Language Model on a dataset of `32,000+` names and generates new names using probability sampling.

The project uses `vv.txt` as the source dataset, copied into:

```text
data/names.txt
```

## Project Overview

This project learns character-to-character transitions from real names.

For example, the name `emma` becomes:

```text
.emma.
```

The model learns pairs:

```text
. -> e
e -> m
m -> m
m -> a
a -> .
```

The `.` token means both start and end of a name. During generation, the model starts at `.` and samples characters until it predicts `.` again.

## Project Structure

```text
bigram_name_generator/
├── data/
│   └── names.txt
├── outputs/
│   ├── generated_names.txt
│   └── bigram_heatmap.png
├── src/
│   ├── __init__.py
│   └── bigram_name_generator.py
├── main.py
├── README.md
└── requirements.txt
```

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

Generate a different number of names:

```bash
python main.py --count 50
```

Use a custom seed:

```bash
python main.py --seed 123
```

## Architecture Diagram

```text
Names Dataset
     |
     v
Character Tokenization
     |
     v
Start/End Token Added
     |
     v
Training Pairs
     |
     v
Bigram Count Matrix
     |
     v
Probability Matrix
     |
     v
Sampling Loop
     |
     v
Generated Names
```

## Data Flow Diagram

```text
Raw name:
    emma

Add boundary tokens:
    .emma.

Create bigrams:
    . -> e
    e -> m
    m -> m
    m -> a
    a -> .

Update matrix:
    count[current_char][next_char] += 1

Convert to probabilities:
    count row / row total

Generate:
    . -> sampled char -> sampled char -> ... -> .
```

## Dataset Analysis

With the provided dataset:

```text
Total names:         32033
Vocabulary size:     27
Longest name:        muhammadibrahim (15 chars)
Average name length: 6.12
```

The vocabulary contains lowercase letters plus the special `.` token:

```text
['.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
 'x', 'y', 'z']
```

## How The Bigram Model Works

A bigram is a pair of neighboring tokens.

In this project, each token is one character. The bigram model asks:

```text
Given the current character, what character usually comes next?
```

So the model learns probabilities like:

```text
P(a | m)
P(n | a)
P(. | a)
```

If many names end in `a`, then `P(. | a)` becomes high. If many names contain `an`, then `P(n | a)` also becomes high.

## Mathematical Explanation

The count matrix stores how often one character follows another.

Let:

```text
C(i, j) = number of times character j appears after character i
```

To convert counts into probabilities:

```text
P(j | i) = C(i, j) / sum(C(i, all_next_characters))
```

This project uses add-one smoothing:

```text
P(j | i) = (C(i, j) + 1) / (sum(C(i, all_next_characters)) + vocabulary_size)
```

Smoothing prevents impossible zero-probability transitions. A transition that never appeared in the dataset can still happen, but rarely.

## Probability Sampling

Generation does not always choose the most likely next character. It samples from the probability distribution.

Example:

```text
Current character: a

Possible next characters:
    n: 0.16
    .: 0.19
    r: 0.09
    l: 0.07
```

Higher-probability characters are more likely, but lower-probability characters can still be selected. This makes generated names more varied.

## Heatmap

The project saves a heatmap here:

```text
outputs/bigram_heatmap.png
```

The heatmap visualizes the probability matrix:

```text
Rows    = current character
Columns = next character
Color   = transition probability
```

Bright cells indicate common transitions.

## Sample Outputs

With seed `42`, the generated names include:

```text
M
Daritajabi
Cilann
Rie
Ya
Asirliyammhndi
Da
Daen
H
Dvin
N
Hylinnla
Eeayolieveedeemon
Ze
A
Mie
Hykyo
Nilan
Jayole
Ttene
```

Some names look realistic, while others look strange. That is expected because the model only remembers one previous character.

## NLP Concepts Learned

- Character-level tokenization
- Vocabulary creation
- Encoding and decoding
- Start and end tokens
- Bigram language modeling
- Count matrices
- Conditional probability
- Add-one smoothing
- Probability sampling
- Text generation
- Matrix visualization

## Limitations Of Bigram Models

Bigram models are simple and useful for learning, but they are very limited.

Main limitations:

- They only look at one previous character.
- They cannot remember the full generated name.
- They cannot learn long-range patterns.
- They may generate names that are too short.
- They may generate names with awkward character sequences.
- They do not understand meaning, gender, culture, or pronunciation.

Example:

```text
If the current character is "a", the model does not know whether the name so far is:

emma
olivia
samantha
alexandra
```

It only knows:

```text
current character = a
```

## Connection Between Bigram Models And GPT

This project and GPT both use the idea of next-token prediction.

Bigram model:

```text
previous character -> next character
```

GPT-style model:

```text
many previous tokens -> next token
```

Similarities:

- Both learn from text data.
- Both tokenize text.
- Both model probability distributions.
- Both generate by repeatedly predicting the next token.

Differences:

- This project uses characters; GPT usually uses subword tokens.
- This project uses one previous character; GPT uses a long context window.
- This project uses a count matrix; GPT uses neural networks.
- This project has no attention; GPT uses Transformer attention.

The bigram model is a tiny conceptual ancestor of modern language models. It teaches the core idea before adding embeddings, neural networks, attention, and Transformers.

## Future Improvements

- Add a trigram model that uses two previous characters.
- Add a neural network model with PyTorch.
- Add temperature control for more or less random generation.
- Filter out very short generated names.
- Score generated names by probability.
- Train on datasets from different languages.
- Build a small web interface with Streamlit or Flask.
- Compare bigram, trigram, and neural models side by side.

---

# Phase 2: Neural Bigram Name Generator

Phase 2 replaces the count matrix with a small PyTorch neural network.

The task is still next-character prediction:

```text
current character -> next character
```

But instead of directly counting transitions, the model learns trainable weights.

## Phase 2 How To Run

```bash
python neural_main.py
```

Run with fewer epochs for a quick test:

```bash
python neural_main.py --epochs 100
```

Predict the next character after a custom prefix:

```bash
python neural_main.py --prompt em
```

The neural bigram model only uses the last character of the prompt. For `em`, it predicts from `m`.

## Phase 2 Outputs

```text
outputs/neural_generated_names.txt
outputs/neural_training_loss.png
```

## Neural Architecture Diagram

```text
Dataset
   |
   v
Tokenization
   |
   v
Encoding
   |
   v
Embedding Layer
   |
   v
Linear Layer
   |
   v
Softmax
   |
   v
Prediction
```

## What The Neural Network Learns

The model receives one character id as input.

Example:

```text
m
```

The character is converted to an integer:

```text
m -> 13
```

The embedding layer converts that id into a learned vector:

```text
13 -> [0.12, -0.44, 0.31, ...]
```

The linear layer converts the vector into one score for every possible next character:

```text
scores for ['.', 'a', 'b', ..., 'z']
```

Softmax converts those scores into probabilities.

## Bigram Count Matrix vs Neural Bigram Model

Count matrix:

```text
Count how often each character follows another character.
Normalize counts into probabilities.
```

Neural model:

```text
Learn embeddings and weights that predict the next character.
Use loss and backpropagation to improve predictions.
```

| Feature | Count Matrix Bigram | Neural Bigram |
|---|---|---|
| Learns by | Counting | Gradient descent |
| Parameters | Count table | Embeddings + linear weights |
| Training loss | No | Yes |
| Optimizer | No | Adam |
| Uses PyTorch | No | Yes |
| Can extend to deep models | Limited | Yes |

## Advantages Of Neural Networks Over Count-Based Models

Neural networks are more flexible because they learn parameters instead of storing only raw counts.

Advantages:

- They can use embeddings.
- They can be trained with loss functions.
- They can be extended with hidden layers.
- They can support larger contexts.
- They are the foundation of modern neural language models.

This specific neural bigram is still simple, but it introduces the training pattern used by larger models.

## Key Learning Concepts

Embedding:

```text
A learned vector representation of a token.
```

Forward propagation:

```text
The input moves through the model to produce predictions.
```

Cross entropy loss:

```text
Measures how wrong the predicted probability distribution is.
```

Backpropagation:

```text
Computes gradients that show how each parameter affected the loss.
```

Gradient descent:

```text
Updates weights in the direction that reduces loss.
```

Adam optimizer:

```text
An adaptive optimizer that often trains neural networks faster and more smoothly than plain gradient descent.
```

## Connection To Modern LLMs

The learning path looks like this:

```text
Bigram Model
      |
      v
Neural Language Model
      |
      v
Attention
      |
      v
Transformer
      |
      v
GPT
```

This project predicts the next character from one previous character.

GPT predicts the next token from many previous tokens.

The shared idea is:

```text
Use previous context to predict the next token.
```

What changes is the power of the model:

- Bigram models use tiny context.
- Neural language models learn trainable representations.
- Attention lets models focus on relevant previous tokens.
- Transformers stack attention and feed-forward layers.
- GPT scales this idea with huge datasets, many parameters, and long context windows.
