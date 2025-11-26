
# Code used on Google Colab
# Here for reference

import pandas as pd
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from google.colab import drive
import numpy as np

# 1. Mount Drive to access YOUR fine-tuned model
drive.mount('/content/drive')

# UPDATE THIS PATH to where your model folder is in Drive
MY_MODEL_PATH = "/content/drive/MyDrive/Cadence2B/model/3_label_class_model_downsampled_2" # Using the model from: /Cadence 2B BTTAI/bert-3_class_label_model-v3/train_loss: 0.587 | val_loss: 0.28 | accuracy: 0.887
print(MY_MODEL_PATH)

# 2. Load Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading user model from: {MY_MODEL_PATH}")

# Load the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(MY_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MY_MODEL_PATH).to(device)

# Ensure device is defined
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SpaCy
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

reviews_path = "/content/drive/MyDrive/Cadence2B/amazon_reviews_2023/balanced_downsampled_reviews_with_3_sentiment_labels.parquet"

if 'tokenizer' not in globals() or 'model' not in globals():
    raise RuntimeError("The 'tokenizer' and 'model' variables are missing. Please run the 'Load Models' cell (Step 2) before running this cell.")

# Print Model Labels to verify mapping
print(f"Model Label Mapping: {model.config.id2label}")

# 3. Data Loading, Cleaning
df = pd.read_parquet(reviews_path)
df = df.head(100000).copy()

def clean_text_for_spacy(text):
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    return text

def get_features(text):
    """Extract features using spaCy (Nouns only) with Clean Filtering"""
    cleaned_text = clean_text_for_spacy(text)
    doc = nlp(cleaned_text)
    features = []

    # Generic nouns to ignore
    ignore_lemmas = {
        'product', 'item', 'amazon', 'review', 'time', 'thing', 'way',
        'lot', 'money', 'issue', 'problem', 'star', 'month', 'year', 'day',
        'purchase', 'order', 'box', 'replacement', 'unit', 'one', 'feature',
        'bit', 'top', 'bottom', 'side', 'part', 'end'
    }

    # Only allow these specific 2-letter words (Tech Acronyms)
    valid_2_letter = {'tv', 'pc', 'cd', 'sd', 'hd', 'vr', 'bt', 'ip', 'os', 'ui', 'av', 'ac', 'dc', 'hq'}

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            # 1. Must be alpha and not a stop word
            if not token.is_stop and token.is_alpha:
                text_lower = token.text.lower()

                # Length Check & Whitelist
                if len(text_lower) < 2:
                    continue

                # If it's exactly 2 letters, it MUST be in our whitelist
                if len(text_lower) == 2 and text_lower not in valid_2_letter:
                    continue

                # Filter out repeated characters (e.g., "aaa", "mmm")
                if len(set(text_lower)) == 1:
                    continue

                lemma = token.lemma_.lower()

                if lemma not in ignore_lemmas:
                    features.append(lemma.capitalize())

    return list(set(features))

def get_sentiment(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Weighted Score Calculation (Assuming [Negative, Neutral, Positive])
    if probs.shape[1] == 3:
        p_neg = probs[0][0].item()
        p_neu = probs[0][1].item()
        p_pos = probs[0][2].item()
        score = (p_neg * 0.0) + (p_neu * 0.5) + (p_pos * 1.0)
    else:
        score = probs[0][1].item()

    return score

# 4. The Processing Loop
results = []

print("Processing reviews...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    text = str(row.get('text', ''))
    if len(text) < 10: continue

    features = get_features(text)
    sentiment_score = get_sentiment(text)

    for feature in features:
        results.append({
            "feature": feature,
            "sentiment_score": sentiment_score
        })

# 5. Aggregation
print("Aggregating data...")
raw_df = pd.DataFrame(results)

dashboard_df = raw_df.groupby("feature").agg(
    mentions=('feature', 'count'),
    sentiment=('sentiment_score', 'mean')
).reset_index()

dashboard_df = dashboard_df[dashboard_df['mentions'] > 5]

save_path = "/content/drive/MyDrive/Cadence2B/amazon_reviews_2023/dashboard_data.csv"
dashboard_df.to_csv(save_path, index=False)
print(f"DONE! Saved to: {save_path}")
print("Sample rows from dashboard data:")

