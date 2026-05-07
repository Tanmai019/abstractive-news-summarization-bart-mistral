# Abstractive News Summarization with BART and Mistral

Transformer-based abstractive news summarization project comparing pretrained and LoRA fine-tuned versions of BART and Mistral on the CNN/DailyMail dataset using ROUGE and BERTScore evaluation metrics.

Developed as part of graduate NLP coursework at the University of Houston, with emphasis on transformer architectures, parameter-efficient fine-tuning, and summarization evaluation.

---

# Tech Stack

`Python` `PyTorch` `Transformers` `Hugging Face` `LoRA` `PEFT` `BART` `Mistral` `NLP` `ROUGE` `BERTScore`

---

# Project Overview

This project explores how modern transformer architectures generate concise and semantically meaningful summaries from long-form news articles. Multiple summarization workflows were implemented and evaluated using both pretrained and parameter-efficient fine-tuned language models.

The project compares:

- Pretrained BART
- LoRA Fine-Tuned BART
- Pretrained Mistral
- LoRA Fine-Tuned Mistral

The evaluation focused on summarization quality, semantic preservation, and generation coherence using standard NLP evaluation metrics.

---

# Dataset

- Dataset: CNN/DailyMail News Summarization Dataset
- Working subset: 3,000 news articles
- Split strategy: 80% training / 20% testing

The dataset contains long-form news articles paired with human-written summaries and is commonly used for benchmarking abstractive summarization systems.

---

# Approach

## 1. Data Preprocessing

The preprocessing pipeline included:

- HTML tag removal
- Unicode normalization
- Text cleaning and whitespace normalization
- Conversion into Hugging Face `Dataset` objects
- Tokenization and sequence preparation

---

## 2. Modeling

### BART Pipeline

Base model:
```python
facebook/bart-large-cnn
```

Key configurations:
- Beam search decoding
- Maximum input length: 1024
- Maximum summary length: 150
- LoRA-based parameter-efficient fine-tuning

The BART workflow leveraged encoder-decoder transformer architecture optimized for abstractive summarization tasks.

---

### Mistral Pipeline

Base model:
```python
mistralai/Mistral-7B-Instruct-v0.2
```

Key configurations:
- Instruction-style prompting
- 4-bit quantization for memory optimization
- LoRA fine-tuning on attention projection layers
- GPU-aware experimentation workflow

The Mistral workflow explored decoder-only LLM summarization behavior under parameter-efficient tuning strategies.

---

# Evaluation Metrics

The generated summaries were evaluated using:

- `ROUGE-1`
- `ROUGE-2`
- `ROUGE-L`
- `ROUGE-Lsum`
- `BERTScore (F1)`

---

# Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | BERTScore (F1) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretrained BART | 0.39 | 0.18 | 0.27 | 0.27 | 0.85 |
| Fine-Tuned BART | 0.40 | 0.19 | 0.29 | 0.29 | 0.87 |
| Pretrained Mistral | 0.12 | 0.07 | 0.09 | 0.09 | 0.82 |
| Fine-Tuned Mistral | 0.17 | 0.08 | 0.12 | 0.15 | 0.83 |

---

# Key Takeaways

- BART achieved the strongest overall summarization performance across all evaluation metrics.
- LoRA fine-tuning produced measurable improvements for both transformer architectures.
- Mistral demonstrated improved semantic generation after fine-tuning, though performance remained below BART.
- Parameter-efficient fine-tuning proved effective for improving summarization quality without requiring full-model retraining.
- Quantization and LoRA workflows enabled experimentation with large language models under limited compute constraints.

---

# Skills Demonstrated

- Transformer-based NLP workflows
- Parameter-efficient fine-tuning (LoRA)
- Summarization model evaluation
- Prompt engineering for LLMs
- GPU-aware experimentation
- Hugging Face ecosystem usage
- Quantization and memory optimization
- Comparative model analysis

---

# Repository Structure

```text
.
├── notebooks/
│   ├── bart/
│   ├── mistral/
│   └── analysis/
├── docs/
│   ├── proposal/
│   └── reports/
├── examples/
├── scripts/
├── requirements.txt
└── README.md
```

---

# Important Files

## Notebooks

- `notebooks/bart/bart_pretrained_and_lora.ipynb`
- `notebooks/mistral/mistral_pretrained.ipynb`
- `notebooks/mistral/mistral_lora_finetuned.ipynb`
- `notebooks/analysis/mistral_fine_tune_graph.ipynb`

---

## Documentation

- Final Report: `docs/reports/final-report.pdf`
- Project Proposal: `docs/proposal/news-summarization-proposal.pdf`

---

## Supporting Files

- Example Input Article: `examples/sample-news-article.docx`
- Notebook Sanitizing Helper: `scripts/sanitize_notebooks.jq`
- Notebook Formatting Helper: `scripts/polish_notebook.jq`

---

# Reproducing the Project

These experiments were developed primarily in GPU-enabled notebook environments such as Google Colab and Kaggle.

## Setup

1. Create a Python environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure Kaggle credentials:

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_key"
```

4. Configure Hugging Face access:

```bash
export HUGGINGFACE_TOKEN="your_hf_token"
```

5. Open the notebooks in Colab, Kaggle, or Jupyter and run the cells sequentially.

---

# Limitations

- Experiments were conducted on a reduced subset due to compute limitations.
- Large-model experimentation introduced GPU memory and runtime constraints.
- Evaluation relied primarily on automated summarization metrics.
- Decoder-only LLM fine-tuning required careful tuning and optimization strategies.

---

# Future Improvements

- Train on larger dataset subsets
- Add human evaluation for factual consistency and fluency
- Benchmark against additional LLM architectures
- Explore retrieval-augmented summarization workflows
- Deploy the best-performing model as an interactive application

---

# Notes

- Notebook outputs and sensitive tokens were removed before publishing.
- The repository preserves the original experimentation workflow while improving structure and readability for portfolio presentation.
- Some notebooks still contain environment-specific Colab/Kaggle utility cells.
