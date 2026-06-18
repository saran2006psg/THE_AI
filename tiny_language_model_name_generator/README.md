# Tiny Language Model and Name Generator

This is a beginner-friendly educational project for learning NLP, deep learning, and LLM fundamentals with Python and PyTorch.

The project builds the same idea in phases:

1. Dataset loading and character tokenization
2. Frequency-based bigram language model
3. Neural network bigram language model
4. Interactive word generation from a starting character
5. Name generation with probability scores

## Setup

From this folder:

```powershell
pip install -r requirements.txt
python tiny_lm_name_generator.py
```

To skip the interactive prompt:

```powershell
python tiny_lm_name_generator.py --no-interactive
```

To train longer:

```powershell
python tiny_lm_name_generator.py --epochs 1000
```

## Project Idea

This project is a tiny version of next-token prediction.

Instead of predicting the next word or subword token like GPT, it predicts the next character in a name.

Example:

```text
e -> m
m -> m
m -> a
a -> .
```

The `.` token means "start" and "end" of a name.

Modern LLMs use the same broad training objective, but with larger datasets, larger vocabularies, many layers, and attention-based Transformer blocks.
