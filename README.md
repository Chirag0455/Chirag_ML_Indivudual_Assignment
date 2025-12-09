# Understanding Attention Mechanisms in Neural Networks  
### A Comparative Tutorial Using RNN, CNN, and Attention-Based Models for Text Classification

This repository contains the full code, figures, and report for a tutorial exploring **attention mechanisms** in neural networks. Through an empirical comparison between an **LSTM**, a **CNN**, and a **BiLSTM with Attention**, this project demonstrates why attention leads to stronger generalisation, better handling of long-range dependencies, and more interpretable decision-making in NLP tasks.

This README accompanies the full report:  

---

## 📌 Project Overview

This tutorial provides:

- A clear introduction to attention mechanisms  
- A comparison between RNN (LSTM), CNN, and Attention architectures  
- Explanation of how attention addresses limitations of traditional sequence models  
- Implementation of three text classification neural networks  
- Performance evaluation on the **IMDb Sentiment Classification** dataset  
- Training and validation accuracy/loss curves  
- Interpretation of model behaviour and generalisation  
- Ethical reflections on explainability and bias  

The goal is to **teach** why attention mechanisms outperform older architectures for many NLP tasks.

---

## 📊 Dataset

**Dataset:** IMDb Movie Review Sentiment Dataset  
- 50,000 labelled reviews  
- Binary sentiment classification (positive / negative)  
- Tokenised and integer-indexed  
- Sequences padded/truncated to length 200  

We load the dataset directly from TensorFlow, so **no manual download is required**.

```python
from tensorflow.keras.datasets import imdb
```

## 🧠 Models Implemented

All models use:

- Embedding layer (64 dimensions)

- Dropout for regularisation

- Same optimiser, batch size, epochs, and train/val splits

1. RNN Model (LSTM)

-- LSTM(64)

-- Dense(32, ReLU)

-- Output layer

2. CNN Model

-- Conv1D(128 filters, kernel size = 5)

-- MaxPooling

-- Conv1D → GlobalMaxPooling

-- Dense layers

3. BiLSTM + Attention Model

-- Bidirectional LSTM(64, return_sequences=True)

-- Custom attention pooling layer

-- Dense layers

This model computes learned attention weights to focus on important tokens.

## 🛠️ How to Run the Notebook
1. Install dependencies
```
pip install tensorflow numpy matplotlib seaborn
```

2. Open the notebook
```
ML&NN_Assign.ipynb
```

3. Run all cells

This will:

- Load IMDb dataset

- Build all three models

- Train for 5 epochs

- Generate training/validation curves

- Evaluate test accuracy and loss

- Save all figures into /figures

## 📁 Folder Structure
```
project/
│
├── figures/
│   ├── Training_Accuracy_comparision.png
│   ├── Training_Loss_comparision.png
│   ├── Validation_Accuracy_comparision.png
│   ├── Validation_Loss_comparision.png
│   └── ...
│
├── ML&NN_Assign.ipynb
├── Chirag_ML_Report_24083279.pdf
├── README.md
└── LICENSE
```

## 📈 Results Summary

From the report’s empirical results:

| Model              | Test Loss  | Test Accuracy |
| ------------------ | ---------- | ------------- |
| RNN (LSTM)         | **0.6566** | **0.5885**    |
| CNN                | **0.5706** | **0.8293**    |
| BiLSTM + Attention | **0.3552** | **0.8479**    |

---

## Key Observations

- LSTM performs poorly due to sequential bottlenecks and difficulty modelling long-range dependencies.

- CNN performs much better by learning local n-gram features efficiently.

- Attention achieves the highest accuracy and lowest loss, due to:

-- capturing global dependencies

-- focusing on emotionally important words

-- improved representational expressiveness

-- providing interpretability through attention weights

These behaviours are consistent with deep learning literature on sequence modelling and attention networks.
---

## 🧠 Why Attention Wins

Based on analysis from the report (Sections 6 & 8):

✔ Overcomes RNN sequential limitations
✔ Captures long-range text relationships
✔ Learns which words matter most for sentiment
✔ Trains faster due to parallel computation
✔ Provides interpretable attention maps
✔ Generalises better than RNNs and CNNs

Attention represents a fundamental shift in how neural networks handle sequences.
---


## 🧩 Ethical Considerations

The report discusses explainability and fairness in NLP:

Attention weights improve transparency but are not perfect explanations

Models may inherit dataset biases

Misclassification in sentiment tasks can reflect linguistic or cultural bias

Attention helps improve robustness, but responsible evaluation remains essential.
---

## 📚 References

All references listed in the report, including:

Bahdanau et al. (2014) — First attention mechanism

Vaswani et al. (2017) — Transformers

Luong et al. (2015) — Attention variants

Hochreiter & Schmidhuber (1997) — LSTM

Kim (2014) — CNN for text

Goodfellow et al. (2016) — Deep Learning textbook
---

## 📄 License

This project is released under the MIT License.
See the LICENSE file for details.
---

## 🙌 Acknowledgements

IMDb dataset creators

TensorFlow/Keras developers

University module instructors

Relevant research authors for foundational work
