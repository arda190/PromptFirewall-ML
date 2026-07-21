# PromptFirewall-ML

Machine Learning-based Prompt Injection Detection for Large Language Models (LLMs)

This project investigates whether classical machine learning
algorithms can detect prompt injection attacks against Large
Language Models.

## Motivation
Large Language Models (LLMs) are vulnerable to prompt injection attacks, where malicious instructions attempt to manipulate the model's behavior or bypass safety mechanisms. Detecting these attacks before they reach an LLM is an important challenge in AI security.

This project investigates whether classical Natural Language Processing (NLP) techniques and machine learning algorithms can accurately identify malicious prompts. Rather than relying solely on modern deep learning models, the project explores interpretable approaches such as TF-IDF, Logistic Regression, and Naive Bayes.

## Objectives

- Detect malicious prompts using machine learning.
- Compare different text vectorization techniques.
- Compare different machine learning classifiers.
- Evaluate model performance using standard metrics.
- Investigate suitable approaches for prompt injection detection.

## Current Features

- Text preprocessing
- Bag of Words (implemented from scratch)
- TF-IDF (implemented from scratch)
- Logistic Regression (implemented from scratch)

## Planned Features

- Naive Bayes
- Support Vector Machine (SVM)
- Word2Vec
- BERT Embeddings
- Performance comparison and evaluation
- Dataset evaluation and benchmarking

## Research Pipeline

```text
Prompt
    ↓
Text Preprocessing
    ↓
Feature Extraction
(Bag of Words / TF-IDF)
    ↓
Machine Learning Classifier
(Logistic Regression / Naive Bayes / SVM)
    ↓
Prediction
(Safe / Prompt Injection)
```

## Future Work
This research aims to compare classical machine learning methods with modern NLP approaches and investigate their applicability to prompt injection detection.

## Implemented from Scratch
This project intentionally implements the following algorithms without using machine learning libraries such as scikit-learn:

- Bag of Words
- TF-IDF
- Logistic Regression
