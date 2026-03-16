# LSTM Next Word Predictor (Metallurgy Text)

This project implements a **Next Word Prediction model using LSTM neural networks** trained on technical metallurgy text extracted from a PDF.
The model learns language patterns from the text and predicts the **most probable next word** given a sequence.

The implementation is built using **TensorFlow / Keras** and basic **NLP preprocessing**.

---

# Project Overview

The goal of this project is to build a **language model** that can generate text by predicting the next word in a sequence.

Pipeline:

1. Extract text from a PDF
2. Clean and preprocess the text
3. Tokenize the text
4. Generate training sequences
5. Train an LSTM neural network
6. Predict the next word iteratively to generate text

The training corpus used here is from:

**George E. Dieter – Mechanical Metallurgy**

---

# Model Architecture

The neural network used:

```
Embedding Layer
      ↓
LSTM Layer (150 units)
      ↓
Dense Layer (Softmax)
```

### Details

| Layer           | Purpose                                          |
| --------------- | ------------------------------------------------ |
| Embedding       | Converts word tokens into dense vectors          |
| LSTM            | Learns sequential language patterns              |
| Dense + Softmax | Outputs probability distribution over vocabulary |

Loss function:

```
categorical_crossentropy
```

Optimizer:

```
Adam
```

---

# Data Preprocessing

### Text Cleaning

The text is cleaned using regex to remove:

* URLs
* citation references
* figure/table labels
* years
* unwanted symbols

### Tokenization

Keras `Tokenizer` is used to convert words into integer tokens.

Example:

```
"The material deforms plastically"
↓
[12, 455, 991]
```

---

# Sequence Generation

A **sliding window** approach is used to generate training samples.

Example sentence:

```
plastic deformation occurs during loading
```

Generated sequences:

```
plastic → deformation
plastic deformation → occurs
plastic deformation occurs → during
plastic deformation occurs during → loading
```

Each sequence is padded so they all have the same length.

---

# Training Data Format

Input:

```
[ w1 , w2 , w3 , w4 ]
```

Target:

```
w5
```

The target word is **one-hot encoded**.

Example:

```
Vocabulary size = 10155
Target vector length = 10155
```

---

# Training

The model is trained with:

```
epochs = 50
validation_split = 0.2
```

Example training shapes:

```
X shape  : (samples, sequence_length)
Y shape  : (samples)
Y onehot : (samples, vocabulary_size)
```

---

# Text Generation

The trained model predicts the next word iteratively.

Example:

```python
test = "theoretical"

for i in range(20):
    tokenized = tokenizer.texts_to_sequences([test])[0]
    padded = pad_sequences([tokenized], maxlen=max_len, padding='pre')
    
    output = model.predict(padded)
    pos = np.argmax(output)
    
    for word,index in tokenizer.word_index.items():
        if index == pos:
            test = test + " " + word
```

This gradually builds a sentence:

```
theoretical stress analysis of deformation behavior ...
```

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/lstm-word-predictor.git
cd lstm-word-predictor
```

Install dependencies:

```
pip install tensorflow pdfplumber numpy
```

---

# Dependencies

* Python 3.x
* TensorFlow / Keras
* NumPy
* pdfplumber
* Regex

---

# Author

**Arindam Choudhary**

Project created for experimentation with **Deep Learning and NLP language models**.
