# Amazon Sentiment Analysis — DistilBERT

> Binary sentiment classification on Amazon product reviews using fine-tuned DistilBERT transformer.

---

## Overview
This project explores the complete NLP pipeline — from exploratory data analysis to transformer-based modeling and deployment — classifying Amazon product reviews as **Positive** or **Negative**.

---

## Project Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| Phase 1 | Problem Definition & Exploratory Data Analysis | ✅ Completed |
| Phase 2 | Traditional ML Baseline — TF-IDF & Classical Classifiers | ✅ Completed |
| Phase 3 | Transformer Model — DistilBERT Fine-tuning | ✅ Completed |
| Phase 4 | Model Comparison & Gradio Deployment | ✅ Completed |

---

## Dataset
| Split | Samples |
|-------|---------|
| Train | 7,000 |
| Validation | 1,496 |
| Test | 1,500 |

- **Source:** Amazon Product Reviews
- **Task:** Binary Classification (Positive / Negative)

---

## Model Configuration
| Parameter | Value |
|-----------|-------|
| Base Model | distilbert-base-uncased |
| Epochs | 3 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Optimizer | AdamW |

---

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 93.27% |
| Precision | 93.27% |
| Recall | 93.27% |
| F1 Score | 93.27% |

---

## Tech Stack
- Python
- HuggingFace Transformers
- PyTorch
- Scikit-learn
- Google Colab
- Gradio

---

## Repository
> This repository contains the transformer-based implementation (Phase 3 & 4).
> Earlier phases are available upon request.

---

*Developed by Eisha · BS Data Science*
