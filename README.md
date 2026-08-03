# PromptFirewall-ML

**A Machine Learning Framework for Detecting Harmful Prompts in Large Language Models**

## Overview

PromptFirewall-ML is a machine learning framework designed to detect harmful prompts before they are processed by a Large Language Model (LLM). The project investigates whether traditional machine learning techniques can effectively identify prompt injection attacks, prompt leakage attempts, jailbreak prompts, and other malicious inputs.

The framework was developed as part of an undergraduate research project and accompanies the paper:

> **PromptFirewall-ML: A Machine Learning Framework for Detecting Harmful Prompts Using Traditional Machine Learning**

---

## Motivation

Large Language Models (LLMs) are increasingly deployed in real-world applications. However, they remain vulnerable to prompt-based attacks that attempt to manipulate model behavior, bypass safety mechanisms, or reveal confidential information.

PromptFirewall-ML explores whether lightweight and interpretable machine learning approaches can detect harmful prompts before they reach an LLM.

---

## Objectives

* Detect harmful prompts using traditional machine learning.
* Compare multiple text feature extraction techniques.
* Evaluate different machine learning classifiers.
* Analyze classification performance using standard evaluation metrics.
* Provide a modular framework for future prompt security research.

---

## Dataset

The framework was evaluated using a custom dataset consisting of **13,831 labeled prompts**:

* **7,731 harmful prompts**
* **6,100 benign prompts**

The dataset contains examples from multiple prompt categories, including:

* Prompt Injection
* Prompt Leakage
* Jailbreak Prompts
* Role-playing Attacks
* Benign User Requests

---

## Preprocessing

The preprocessing pipeline includes:

* Lowercase conversion
* Punctuation removal
* Whitespace normalization
* Stop-word removal

---

## Feature Extraction

The following feature extraction techniques are supported:

### Implemented from Scratch

* Bag of Words (BoW)
* TF-IDF

### Library-Based Implementations

* Word2Vec (Gensim)
* FastText (Gensim)

---

## Machine Learning Models

### Implemented from Scratch

* Multinomial Naive Bayes
* Logistic Regression

### Library-Based Implementation

* Support Vector Machine (scikit-learn)

> **Note:** The current experimental results include Naive Bayes and Support Vector Machine. The custom Logistic Regression implementation is included in the repository and will be evaluated in future work.

---

## Evaluation Metrics

The framework evaluates model performance using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## Experimental Pipeline

```text
Dataset
    │
    ▼
Text Preprocessing
    │
    ▼
Feature Extraction
 ├── Bag of Words
 ├── TF-IDF
 ├── Word2Vec
 └── FastText
    │
    ▼
Machine Learning Classifier
 ├── Naive Bayes
 ├── Logistic Regression
 └── Support Vector Machine
    │
    ▼
Prediction
(Harmful / Benign)
```

---

## Project Structure

```text
PromptFirewall-ML/
│
├── dataset/
├── vectorizers/
├── models/
├── preprocessing/
├── pipeline/
├── experiment/
├── evaluation/
├── run_experiments.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/arda190/PromptFirewall-ML.git
cd PromptFirewall-ML
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

Execute:

```bash
python run_experiments.py
```

The framework will preprocess the dataset, generate feature vectors, train the selected classifier, and report the evaluation metrics.

---

## Future Work

Future improvements include:

* Evaluation of the custom Logistic Regression implementation
* BERT and transformer-based embeddings
* Larger and more diverse datasets
* Cross-validation experiments
* Hyperparameter optimization
* Comparison with deep learning models
* Evaluation on public benchmark datasets

---

## Technologies

* Python
* NumPy
* Pandas
* scikit-learn
* Gensim

---

## Author

**Arda Karakaş**

Department of Computer Engineering

Izmir University of Economics

---

## License

This project is intended for educational and research purposes.

