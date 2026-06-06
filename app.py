import gradio as gr
import torch
from transformers import (DistilBertTokenizer,
                          DistilBertForSequenceClassification)

# Load model from Hugging Face
model_name = "EishaSohail/amazon-sentiment-model"

tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model     = DistilBertForSequenceClassification.from_pretrained(model_name)
model.eval()

def predict_sentiment(review_text):
    if not review_text.strip():
        return "Please enter a review!"

    inputs = tokenizer(
        review_text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs    = torch.softmax(outputs.logits, dim=1)
    pred     = torch.argmax(probs, dim=1).item()
    neg_prob = probs[0][0].item() * 100
    pos_prob = probs[0][1].item() * 100

    if pred == 1:
        return f"""✅ POSITIVE REVIEW

Confidence:
- Positive : {pos_prob:.1f}%
- Negative : {neg_prob:.1f}%"""
    else:
        return f"""❌ NEGATIVE REVIEW

Confidence:
- Positive : {pos_prob:.1f}%
- Negative : {neg_prob:.1f}%"""

app = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Type your Amazon review here...",
        label="Enter Review"
    ),
    outputs=gr.Textbox(
        label="Prediction",
        lines=6
    ),
    title="🛍️ Amazon Review Sentiment Analyzer",
    description="Powered by DistilBERT — 93.27% Accuracy",
    examples=[
        ["This product is absolutely amazing! Best purchase ever!"],
        ["Terrible quality, broke after one day. Waste of money."],
        ["It works okay, nothing special but does the job."]
    ],
    theme=gr.themes.Soft()
)

app.launch()