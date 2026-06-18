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
