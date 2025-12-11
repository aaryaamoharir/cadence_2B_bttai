# 1. Import libraries
import pandas as pd
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import numpy as np

print("hi")

# 2. Load your data
import os
path = "/content/sample_data/balanced_reviews.parquet"
print(os.path.exists(path), os.path.getsize(path))

df = pd.read_parquet("/content/sample_data/balanced_reviews.parquet")
file_path = '/content/sample_data/stratified_electronics_metadata.parquet'
df_feature = pd.read_parquet(file_path)

# After merging, handle missing descriptions properly
merged_df = pd.merge(df, df_feature, on='parent_asin', how='right')

# Handle missing/empty descriptions
merged_df['description'] = merged_df['description'].fillna('')

# If description is a list or array, join it into a string
def clean_description(desc):
    # Handle numpy arrays
    if isinstance(desc, np.ndarray):
        if desc.size == 0:
            return ''
        return ' '.join(str(x) for x in desc if x)
    
    # Handle lists
    if isinstance(desc, list):
        if not desc:
            return ''
        return ' '.join(str(x) for x in desc if x)
    
    # Handle None, empty strings, and literal string representations
    if pd.isna(desc):
        return ''
    
    desc_str = str(desc).strip()
    if desc_str in ('', '[]', 'nan', 'None'):
        return ''
    
    return desc_str

merged_df['description'] = merged_df['description'].apply(clean_description)

# Truncate very long descriptions for speed
merged_df['description'] = merged_df['description'].str[:1000]

# Filter out rows with empty descriptions to avoid wasted processing
merged_df_with_text = merged_df[merged_df['description'].str.len() > 10].copy()

print(f"Processing {len(merged_df_with_text)} samples with descriptions...")
print(f"Skipped {len(merged_df) - len(merged_df_with_text)} samples with no/short descriptions")

# Show sample descriptions to verify
print("\nSample descriptions:")
for i, desc in enumerate(merged_df_with_text['description'].head(3)):
    print(f"{i}: {desc[:200]}...")

# ⚡ Reduce dataset size for testing (only 50 samples)
merged_df_with_text = merged_df_with_text.head(50).copy()
print(f"Processing {len(merged_df_with_text)} samples...")

# 3. Load NLP and transformer models (optimized)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load spaCy with only needed components
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
nlp.max_length = 2000000

model_absa = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_absa)
model = AutoModelForSequenceClassification.from_pretrained(model_absa).to(device)

print("hi3")

# 4. Define BATCHED aspect-based sentiment analysis function
def analyze_aspects_batch(texts, nlp, tokenizer, model, device, batch_size=16, max_aspects=15):
    """Process multiple texts at once with batching"""
    all_results = []
    
    # Extract aspects for all texts first
    all_aspects = []
    for text in texts:
        doc = nlp(text)
        aspects = []
        for sent in doc.sents:
            if len(aspects) >= max_aspects:  # Limit aspects per description
                break
            for chunk in sent.noun_chunks:
                if chunk.root.pos_ not in ["NOUN", "PROPN"]:
                    continue
                if len(chunk.text.split()) == 1 and chunk.root.tag_ in ["PRP", "PRP$"]:
                    continue
                aspect = chunk.text.strip()
                aspects.append((aspect, sent.text.strip()))
        all_aspects.append(aspects)
    
    # Batch sentiment analysis
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    
    for aspects in all_aspects:
        if not aspects:
            all_results.append([])
            continue
        
        results = []
        # Process aspects in batches
        for i in range(0, len(aspects), batch_size):
            batch_aspects = aspects[i:i+batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                [f"{sent} [SEP] {asp}" for asp, sent in batch_aspects],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                sentiment_idx = logits.argmax(dim=-1)
                probs = torch.softmax(logits, dim=-1)
                
                for j, (aspect, _) in enumerate(batch_aspects):
                    label = sentiment_labels[sentiment_idx[j].item()]
                    confidence = probs[j, sentiment_idx[j]].item()
                    results.append((aspect, label, confidence))
        
        all_results.append(results)
    
    return all_results

# 5. Process in batches with progress bar
text_batch_size = 8  # Process 8 descriptions at a time
all_results = []

for i in tqdm(range(0, len(merged_df_with_text), text_batch_size), desc="Processing batches"):
    batch_texts = merged_df_with_text['description'].iloc[i:i+text_batch_size].tolist()
    try:
        batch_results = analyze_aspects_batch(
            batch_texts, nlp, tokenizer, model, device, 
            batch_size=16,  # Sentiment analysis batch size
            max_aspects=15   # Max aspects per description
        )
        all_results.extend(batch_results)
    except Exception as e:
        print(f"Error at batch starting index {i}: {e}")
        all_results.extend([[] for _ in range(len(batch_texts))])

merged_df_with_text['aspect_sentiments'] = all_results

# 6. Inspect results
print(merged_df_with_text[['description', 'aspect_sentiments']].head())
print(f"\nTotal aspects extracted: {sum(len(x) for x in all_results)}")

# 7. Display results in a readable format
print("\n" + "="*80)
print("DETAILED RESULTS")
print("="*80)

for idx in range(min(5, len(merged_df_with_text))):  # Show first 5 examples
    row = merged_df_with_text.iloc[idx]
    
    print(f"\n{'='*80}")
    print(f"PRODUCT {idx + 1}")
    print(f"{'='*80}")
    
    # Show description
    print(f"\nDESCRIPTION:")
    print(f"{row['description'][:500]}...")  # First 500 chars
    
    # Show extracted aspects and sentiments
    print(f"\nEXTRACTED FEATURES & SENTIMENTS:")
    print(f"{'Feature':<40} {'Sentiment':<12} {'Confidence'}")
    print("-" * 80)
    
    if row['aspect_sentiments']:
        for aspect, sentiment, confidence in row['aspect_sentiments']:
            print(f"{aspect:<40} {sentiment:<12} {confidence:.3f}")
    else:
        print("(No features extracted)")
    
    print()

# 8. Create a more detailed DataFrame view
expanded_rows = []
for idx, row in merged_df_with_text.iterrows():
    if row['aspect_sentiments']:
        for aspect, sentiment, confidence in row['aspect_sentiments']:
            expanded_rows.append({
                'description': row['description'][:100] + '...',  # Truncate for display
                'feature': aspect,
                'sentiment': sentiment,
                'confidence': confidence
            })
    else:
        expanded_rows.append({
            'description': row['description'][:100] + '...',
            'feature': 'No features',
            'sentiment': 'N/A',
            'confidence': 0.0
        })

results_df = pd.DataFrame(expanded_rows)

print("\n" + "="*80)
print("SUMMARY TABLE (First 20 rows)")
print("="*80)
print(results_df.head(20).to_string(index=False))

# 9. Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Total products processed: {len(merged_df_with_text)}")
print(f"Total features extracted: {len(results_df)}")
print(f"Average features per product: {len(results_df) / len(merged_df_with_text):.2f}")
print(f"\nSentiment distribution:")
print(results_df['sentiment'].value_counts())
print(f"\nAverage confidence: {results_df['confidence'].mean():.3f}")