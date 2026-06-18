# Character Predictor with Python and PyTorch

This beginner-friendly project trains a tiny character-level neural network to
predict the next character from previous characters.

## Run

```powershell
cd character_predictor
python char_predictor.py
```

To run the next-sentence version:

```powershell
python sentence_predictor.py
```

If PyTorch is not installed:

```powershell
pip install torch
```

## What the Project Teaches

### 1. What is tokenization?

Tokenization is the process of breaking text into smaller pieces called tokens.
In this project, each character is one token.

Example:

```text
"hello" -> ["h", "e", "l", "l", "o"]
```

Large language models usually use subword tokens instead of single characters,
but the idea is the same: text must be split into model-readable units.

### 2. What is vocabulary?

A vocabulary is the complete set of tokens the model knows.

For character-level modeling, the vocabulary is the set of unique characters in
the dataset.

Example:

```text
Text: "hello"
Vocabulary: ["e", "h", "l", "o"]
```

### 3. Why is encoding needed?

Neural networks work with numbers, not raw text. Encoding converts each token
into an integer ID.

Example:

```text
h -> 1
e -> 0
l -> 2
o -> 3
```

Then:

```text
"hello" -> [1, 0, 2, 2, 3]
```

### 4. Why are embeddings used?

An embedding layer converts integer IDs into learned vectors.

Instead of treating each character as just a number, embeddings let the model
learn useful numeric representations for characters. During training, characters
that appear in similar contexts can develop useful patterns in their vectors.

### 5. What does Cross Entropy Loss do?

Cross Entropy Loss measures how different the model's predicted probabilities
are from the correct answer.

If the correct next character is `"o"` and the model gives `"o"` a high
probability, the loss is low. If the model gives `"o"` a low probability, the
loss is high.

### 6. What does Adam optimizer do?

Adam is an optimizer that updates the model's weights during training. It uses
the gradients from backpropagation and adapts the step size for each parameter.

This often makes training smoother and faster than basic gradient descent.

### 7. How does backpropagation update weights?

Backpropagation calculates how much each weight contributed to the error.

Training follows this loop:

1. Make a prediction.
2. Calculate the loss.
3. Compute gradients using `loss.backward()`.
4. Update weights using `optimizer.step()`.
5. Repeat many times.

Over time, the weights change so the model assigns higher probabilities to
better next-character predictions.

### 8. How does next-character prediction relate to GPT and LLMs?

GPT-style models are trained with the same basic idea: predict what comes next.

This project predicts the next character. GPT predicts the next token, where a
token may be a word, part of a word, punctuation mark, or symbol.

The big differences are scale and architecture:

- This project uses a tiny dataset and a simple neural network.
- GPT uses huge datasets and transformer neural networks.
- This project predicts from a short fixed window.
- GPT can use much longer context windows.

Even so, the learning objective is closely related: use previous tokens to
predict the next token.

## Next-Sentence Version

The file `sentence_predictor.py` uses the same learning ideas, but predicts the
next sentence instead of the next character.

It treats each line in `sentence_data.txt` as one sentence and creates samples
like this:

```text
"Text can be split into smaller tokens."
    -> "Tokens are converted into integer ids."
```

This model does not generate new sentences from scratch. It chooses the most
likely next sentence from the sentences it saw in the training dataset.
