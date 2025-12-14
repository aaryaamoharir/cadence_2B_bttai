
# Breakthrough Tech AI Studio Project Fall 2025 - Cadence2B 



### 👥 **Team Members**


| Name             | GitHub Handle | Contribution                                                             |
|------------------|--------------- |--------------------------------------------------------------------------|
| Aaryaa Moharir   | @aaryaamoharir | Data exploration, feature extraction, ABSA model integration, dashboard  |
| Hannah Rauch     | @hannahrauch     | Data collection, data cleaning, exploratory data analysis (EDA), preliminary sentiment analysis, keyword and product feature extraction, data visualization, model integration  |
| Darlyn Gomez     | @aminahassan  | Data preprocessing, feature engineering, data validation                 |
| Jianhua Deng     | @Jianhua-Deng      | Model selection, hyperparameter tuning, model training and optimization  |

---

## 🎯 **Project Highlights**

- Developed a machine learning pipeline using natural language processing, feature extraction, and aspect-based sentiment analysis (ABSA) to help Cadence product designers identify which software features users like or dislike based off of the Amazon 2023 reviews dataset. 
- Conducted baseline experiments using BERT and feature extraction models, identifying early accuracy plateaus (55–63%) and designing strategies to improve training efficiency.
- Performed extensive exploratory data analysis (EDA), uncovering noise, inconsistent formatting, and ambiguous feature mentions in product reviews—informing preprocessing and modeling decisions. For example, discovering class imbalance between different ratings. 
- Generated actionable insights to inform business decisions at Cadence using a Streamlit dashboard.
- Implemented the ABSA Model(Aspect-based Sentiment Analysis Model) to address industry needs around fine-grained user sentiment insights.
  
---

## 👩🏽‍💻 **Setup and Installation**

* How to clone the repository: git clone "" 
* How to install dependencies: pip install -r requirements.txt
* How to set up the environment:
    1. python3 -m venv venv
    2. source env/bin/activate
* How to access the dataset(s): navigate to this page https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 
* How to run the notebook or scripts:
     1. To run the streamlit dashboard: streamlit run dashboard.py
     2. To run the python models: python [nameOfFile.y]
---

## 🏗️ **Project Overview**

This project was completed as part of the Break Through Tech AI Studio, where teams work with industry partners to deliver machine learning solutions to real-world problems.

Our host company, Cadence Design Systems, asked us to analyze product reviews to better understand:

1. Which software features users discuss most
2. Sentiment associated with each feature
3. Areas of improvement and opportunities for customer satisfaction

The goal of the project is to design an aspect-based sentiment analysis model capable of extracting fine-grained insights that traditional sentiment analysis cannot provide. This work has real-world significance for user experience design, product engineering prioritization, and customer-driven innovation.

---

## 📊 **Data Exploration**

We used the Amazon Reviews 2023 dataset, consisting of approximately 43 million customer reviews containing review text, star ratings, timestamps, and product metadata. To make the project computationally feasible and domain-relevant, the dataset was filtered to the Electronics category, resulting in approximately 1 million reviews. The data was stored and processed in parquet format for efficient access and scalability


*Data Exploration & Preprocessing:*

We performed extensive preprocessing to prepare the dataset for modeling:
    -Removed HTML tags, duplicate entries, and extremely short or low-quality (“garbage”) reviews
    -Cleaned text by removing filler words, conjunctions, and stopwords
    -Identified and addressed significant class imbalance, as the majority of reviews were overwhelmingly positive
    -Applied random sampling to reduce dataset size and accelerate experimentation without losing representativeness

*Exploratory Data Analysis (EDA) Insights*

Key insights from EDA included:
    -Strong positive bias in review sentiment across most products
    -Clear differences in review volume and sentiment patterns by product type
    -Frequent mentions of specific product features such as battery life, usability, camera quality, and screen
    -Evidence that star ratings alone fail to capture feature-level dissatisfaction (e.g., high rating but negative battery comments)

*Challenges:*

Severe sentiment class imbalance
Long model runtimes when processing large-scale text data
Noisy feature extraction when relying solely on rule-based NLP approaches (e.g., spaCy noun extraction)

*Assumptions:*

Reviews accurately reflect user sentiment toward specific product features
Sampling preserved overall sentiment and feature distributions
Pretrained language models generalize well to Amazon review data

---

## 🧠 **Model Development**

We initially explored separate sentiment classification and feature extraction pipelines, but later pivoted to an Aspect-Based Sentiment Analysis (ABSA) approach that combines both tasks.

*Final model:*

    -Pretrained DeBERTa / BERT-based transformer model
    -Loaded from Hugging Face
    -Fine-tuned for sentiment classification at the feature (aspect) level

*Feature Selection & Tuning:*

    -Feature (aspect) extraction using spaCy-based parsing
    -Input pairs constructed as (review text, extracted feature)
    -Tokenization via AutoTokenizer
    -Sentiment classification using AutoModelForSequenceClassification
    -Leveraged transfer learning to avoid training from scratch and significantly reduce compute time


---

## 📈 **Results & Key Findings**

*Performance Metrics:*

    -Sentiment classification accuracy: ~88%
    -Processing time reduced from days to hours using pretrained transformer models 

*Model Performance & Insights:*

    -Successfully extracted feature-level sentiment, solving the “mixed-review” problem
    -Enabled outputs such as:
        Camera sentiment: 4.8 / 5
        Battery sentiment: 2.1 / 5
    -Demonstrated that high star ratings can mask dissatisfaction with specific product features
    
*Limitations:*

    -The dataset’s positivity bias may skew sentiment predictions
    -Certain extracted “features” were noisy or irrelevant (e.g., “works,” “weeks”)
    -Feature extraction accuracy was limited when relying only on spaCy without domain-specific constraints

---

## 🚀 **Next Steps**

*Current Limitations:*

    -Feature extraction quality still produces some irrelevant aspects
    -Class imbalance limits robustness for negative sentiment detection
    -Sampling trades completeness for speed

*Future Improvements:*

    -Improve feature extraction using domain-aware or supervised aspect models
    -Perform deeper hyperparameter tuning
    -Add sentiment calibration across product categories
    -Incorporate multilingual reviews
    -Scale inference using distributed processing

---

## 🙏 **Acknowledgements** 

Thank you Dr. Farhan Raseed, Matt Brems, and the entire Break Through Tech team for helping us throughout this process and giving us the opporutnity to build this project!

