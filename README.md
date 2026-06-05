readme = """
# Sentiment Analysis on Amazon Product Reviews

## Project Overview
Fine-tuned DistilBERT transformer model to classify 
Amazon product reviews as Positive or Negative.

## Dataset
- Name     : Amazon Product Reviews
- Source   : Kaggle
- Link     : https://www.kaggle.com/datasets/
             kritanjalijain/amazon-reviews
- Size     : 400,000+ reviews
- Columns  : Review (text), Label (0/1)
- Classes  : 0 = Negative | 1 = Positive
- Train    : 7000 | Val: 1496 | Test: 1500

## Label Mapping
Original Dataset:
- Label 1 = Neutral  → Remapped to 0 (Negative)
- Label 2 = Positive → Remapped to 1 (Positive)

## Model
- Name          : DistilBERT
- Version       : distilbert-base-uncased
- Epochs        : 3
- Batch Size    : 16
- Learning Rate : 2e-5
- Max Length    : 128 tokens

## Results
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 93.27% |
| Precision | 93.27% |
| Recall    | 93.27% |
| F1 Score  | 93.27% |

## Project Structure
- Assignment 1 → Problem Definition & EDA
- Assignment 2 → Traditional ML (TF-IDF)
- Assignment 3 → Transformer Model (DistilBERT)
- Assignment 4 → Comparison & Deployment

## Tools Used
- Python 3
- HuggingFace Transformers
- PyTorch
- Google Colab (T4 GPU)
- Scikit-learn
- Pandas
- Gradio (Deployment)

## How To Run
1. Open notebook in Google Colab
2. Enable T4 GPU
3. Run all cells sequentially
"""

with open('/content/nlp-sentiment-analysis/README.md', 'w') as f:
    f.write(readme)

# Push to GitHub
%cd /content/nlp-sentiment-analysis
!git add .
!git commit -m "Updated README with Kaggle dataset info"
!git push

print("README updated & pushed ✅")
