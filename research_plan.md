# PromptFirewall-ML: Low-Latency, Multi-Layered Prompt Injection Detection Gateway

## 📌 Abstract / Summary
Large Language Models (LLMs) are increasingly deployed in real-world API workflows, making them targets for **Prompt Injection Attacks** (Direct Jailbreaks, Indirect Injections, and System Prompt Leaks). Current commercial safeguards rely on heavy secondary LLMs, introducing unacceptable API latency and cost overheads. 

**PromptFirewall-ML** investigates the trade-offs between inference speed, resource consumption, and detection accuracy by benchmarking classical NLP feature extraction techniques against modern lightweight transformer embeddings within a multi-tiered security gateway.

---

## 🎯 Key Research Objectives
1. **Low-Latency Gateway Architecture:** Achieve sub-50ms detection latencies suitable for real-time API filtering without degrading LLM user experience.
2. **Feature Extraction Comparison:** Evaluate sentence-level representation capabilities across **TF-IDF, Word2Vec, FastText, and DistilBERT** under adversarial prompt perturbations.
3. **From-Scratch Baseline Benchmarking:** Implement foundational mathematical models from scratch (Logistic Regression, Naive Bayes, TF-IDF) to establish zero-dependency latency and memory overhead baselines before benchmarking against PyTorch/Scikit-Learn implementations.
4. **False Positive Minimization:** Benchmark models against complex benign prompts (technical code blocks, creative writing) to ensure False Positive Rates (FPR) remain under 2%.

---

## 🔬 System Pipeline Architecture

```text
[ Incoming User Prompt ]
          │
          ▼
┌───────────────────────────────────┐
│ Tier 1: Static Heuristics & Rules │  ──(High Confidence Attack)──► [ Block Request ]
└───────────────────────────────────┘
          │ (Pass / Ambiguous)
          ▼
┌───────────────────────────────────┐
│ Tier 2: Preprocessing & Cleaning │  (Tokenization, Lowercasing, Obfuscation Handling)
└───────────────────────────────────┘
          │
          ▼
┌───────────────────────────────────┐
│ Tier 3: Feature Vectorization     │  (TF-IDF / Word2Vec / DistilBERT Embeddings)
└───────────────────────────────────┘
          │
          ▼
┌───────────────────────────────────┐
│ Tier 4: ML Classification Layer   │  (Logistic Regression / Naive Bayes / SVM)
└───────────────────────────────────┘
          │
  ┌───────┴───────┐
  ▼               ▼
[ Safe Prompt ]  [ Injection Detected ]
  │               │
  ▼               ▼
(Forward to LLM) (Return 403 Forbidden)





