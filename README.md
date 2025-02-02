# Reality Weaver (TRW) – An Innovative AI Approach for the Greater Benefit of Humanity 🧘‍♂️🚀

**Reality Weaver (TRW)** is a groundbreaking project developed as part of [Owlchemy.xyz](https://owlchemy.xyz) that seeks to harness artificial intelligence to process ancient wisdom texts and transform them into meaningful representations for deep learning. This innovative approach is designed to help humanity gain insights into the nature of reality through the lens of Yogavasista wisdom.

## 📚 Overview

TRW processes raw affirmation texts from three distinct phases of wisdom:
- **Fumigation** – The cleansing of the mind.
- **Awakening** – The process of gaining insight.
- **Enlightenment** – Transcending limitations to understand the ultimate nature of reality.

The pipeline includes:
- **Text Processing:** Cleansing, tokenization, and numerical encoding of text.
- **Learnable Phi‑Layer:** A unique transformation inspired by Yogavasista philosophy, combining non-linear functions (tanh and sin) with learnable parameters to enrich embeddings.
- **Text Classifier Model:** A simple classifier that aggregates token embeddings and produces output logits.
- **Background Training & Interactive Inference:** Simulated continuous training and an interactive query loop for real‑time model predictions.

## 🛠️ How It Works

1. **Data Loading & Processing:**  
   The raw text is loaded from files, cleaned (removing punctuation and extra spaces), tokenized, and then numerically encoded. Affirmations are labeled by phase based on their source file.

2. **Learnable Phi‑Layer:**  
   This layer transforms the embeddings using the formula:
   > φ(x) = α · tanh(x) + β · sin(x) + γ  
   where **α**, **β**, and **γ** are learnable parameters.

3. **Embedding & Classification:**  
   The processed token IDs are converted into embeddings, transformed by the phi‑layer, and then aggregated to form a representation for classification via a linear layer.

4. **Interactive Inference:**  
   Users can input queries, which are processed and passed through the model to output prediction logits.

## 🚀 Setup and Running

**Prerequisites:**  
- Rust (latest stable version)  
- The text files: `fumi_corpus.txt`, `awak_corpus.txt`, and `enlight_corpus.txt` placed in the appropriate folder.

**Build and Run:**

```bash
cargo build
cargo run
