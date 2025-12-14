# Breakthrough Tech AI Studio Project Fall 2025 | Cadence2B

### 👥 **Team Members**

| Name           | GitHub Handle  | Contribution                                                                                                                                                                                             |
| -------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Aaryaa Moharir | @aaryaamoharir | Data exploration, feature extraction, ABSA model integration, dashboard                                                                                                                                  |
| Hannah Rauch   | @hannahrauch   | Data collection, data cleaning, exploratory data analysis (EDA), preliminary sentiment analysis, keyword and product feature extraction, data visualization, model integration                           |
| Darlyn Gomez   | @DarlynGomez   | Exploratory data analysis (EDA), machine learning architecture & modeling, hyperparameter tuning, ABSA pipeline development, generative AI experimentation, production optimization, and model selection |
| Jianhua Deng   | @Jianhua-Deng  | Model selection, hyperparameter tuning, model training and optimization                                                                                                                                  |

---

## ★ **Project Highlights**

- Developed a **machine learning pipeline** using natural language processing, feature extraction, and aspect-based sentiment analysis (ABSA) to help Cadence product designers identify which software features users like or dislike based off of the Amazon 2023 reviews dataset.
- Performed extensive **exploratory data analysis (EDA)**, uncovering noise, inconsistent formatting, and ambiguous feature mentions in product reviews informing preprocessing and modeling decisions. For example, discovering class imbalance between different ratings.
- Built **interactive Streamlit dashboard** with filtering across 1M+ reviews
- Conducted **baseline experiments** using BERT and feature extraction models, identifying early accuracy plateaus (55-63%) and pivoted to integrated ABSA approach
- Experimented with **generative AI models** (Google Gemini, DeepSeek) achieving 200K review processing in 25 minutes with AI-generated actionable insights
- Implemented the **ABSA Model**(Aspect-based Sentiment Analysis Model) to address industry needs around fine-grained user sentiment insights.
- Optimized **processing pipeline** from 3-day runtime to hours through model architecture redesign

---

## ➤ **Setup and Installation**

### ⓘ Large Data Files

Due to GitHub's file size limits, the following file is not included in the repository:

- **aspect_sentiments.csv** (ABSA predictions, ~217 MB)
  - Contains 1M+ aspect-level sentiment predictions
  - Required to run the dashboard
  - [Download from Google Drive](https://drive.google.com/file/d/1eET8j8Bt96xqe0NWZuqS43FL_HAW_Mkl/view?usp=sharing)
  - Place in `data/` directory after cloning

### To run the dashboard:

1. Clone this repository
2. Download aspect_sentiments.csv from the link above
3. Place it in the `data/` directory
4. Run `streamlit run dashboard/ultimate_dashboard.py`

- How to clone the repository: `git clone "https://github.com/aaryaamoharir/cadence_2B_bttai.git"`
- How to install dependencies: `pip install -r requirements.txt`
- How to set up the environment:
  1. `python3 -m venv venv`
  2. `source venv/bin/activate`
- How to access the dataset(s): Navigate to https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- How to run the project:
  1.  To run the Streamlit dashboard: `streamlit run dashboard.py`
  2.  To run the Python models: `python [nameOfFile.py]`
  3.  To view ABSA results: Open `aspect_sentiments.csv` or explore via dashboard

---

## ⟡ **Project Overview**

This project was completed as part of the **Break Through Tech AI Studio**, where teams collaborate with industry partners to deliver machine learning solutions for real-world problems.

Our host company, **Cadence Design Systems**, tasked us with analyzing product reviews to understand:

1. Which software features users discuss most
2. The sentiment associated with each feature
3. Areas for improvement and opportunities to enhance customer satisfaction

The project goal was to design an **aspect-based sentiment analysis (ABSA) model** capable of extracting fine-grained insights that traditional sentiment analysis cannot capture. This work has practical implications for **user experience design**, **product engineering prioritization**, and **customer-driven innovation**.

### **Our Solution**

We developed an **ABSA system** that:

- Extracts specific product aspects such as battery, screen, and camera
- Assigns independent sentiment scores for each aspect
- Provides actionable insights, for example:

  - **Camera quality:** 4.8 / 5 sentiment
  - **Battery life:** 2.1 / 5 sentiment

This approach goes beyond generic sentiment summaries, offering detailed, feature-level intelligence that supports informed product decisions.

---

## ⟡ **Data Exploration**

We used the Amazon Reviews 2023 dataset, which contains approximately 43 million customer reviews including review text, star ratings, timestamps, and product metadata. For computational feasibility and domain relevance, the dataset was filtered to the Electronics category, resulting in around 1 million reviews. Data was stored and processed in **parquet format** for efficient access and scalability.

### **Data Exploration & Preprocessing**

- Removed HTML tags, duplicate entries, and extremely short or low-quality reviews
- Cleaned text by removing filler words, conjunctions, and stopwords
- Addressed significant class imbalance, as most reviews were overwhelmingly positive
- Applied stratified random sampling to reduce dataset size while maintaining representativeness

### **Exploratory Data Analysis (EDA) Insights**

- Review sentiment shows strong positive bias across most products
- Differences in review volume and sentiment patterns are apparent by product type
- Frequent mentions of key product features: battery life, usability, camera quality, and screen
- Star ratings alone do not capture feature-level dissatisfaction (for example, 5-star ratings may include negative battery comments)
- Over 1,000 rating-sentiment disagreements detected where numerical ratings contradict text sentiment

### **Challenges**

- Severe class imbalance in sentiment distribution
- Long runtimes when processing large-scale text data (initial approach took 3 days)
- Noisy feature extraction from rule-based NLP methods (e.g., spaCy extracting generic terms like "weeks," "works," "things")
- Mixed-sentiment reviews with positive and negative comments about different features

### **Assumptions**

- Reviews accurately reflect user sentiment toward specific product features
- Stratified sampling preserved overall sentiment and feature distributions
- Pretrained language models generalize effectively to Amazon review data

---

## ⟡ **Model Development**

We initially explored separate sentiment classification and feature extraction pipelines using **BERT** and **spaCy**, but later adopted an **Aspect-Based Sentiment Analysis (ABSA)** approach that integrates sentiment classification and feature extraction into a single model.

### **Final Model**

- **Model:** `yangheng/deberta-v3-base-absa-v1.1` from Hugging Face
- **Architecture:** Pre-trained DeBERTa v3 transformer fine-tuned for aspect-level sentiment
- **Key Innovation:** Identifies product aspects and assigns independent sentiment to each aspect in a single pass
- **Accuracy:** Maintains 88% sentiment classification
- **Processing Time:** Reduced from 3 days to hours
- **Advantage:** Effectively handles mixed-sentiment reviews, e.g., "great camera but terrible battery" is correctly classified as **camera = positive**, **battery = negative**

### **Feature Selection & Tuning**

- Leveraged **transfer learning** to avoid training from scratch, reducing compute requirements
- Used a pre-trained checkpoint fine-tuned on millions of product reviews
- Constructed **review-aspect pairs** to enable contextualized sentiment analysis
- Applied **batch processing optimization** to ensure scalability

---

## ★ **Results & Key Findings**

### **Performance Metrics**

- **Sentiment classification accuracy:** ~88%
- **Processing time:** Reduced from days to hours through the use of pretrained transformer models

### **Model Performance & Insights**

- Successfully extracted **feature level sentiment**, addressing the mixed review problem
- Enabled granular outputs such as:

  - **Camera sentiment:** 4.8 / 5
  - **Battery sentiment:** 2.1 / 5

- Demonstrated that high overall star ratings can mask dissatisfaction with specific product features

---

## ⟡ **Next Steps**

### **Current Limitations**

- Dataset positivity bias may skew sentiment predictions toward positive classes
- Feature extraction using spaCy alone produced noisy or generic features, such as "works" and "weeks"
- Class imbalance reduced robustness when detecting infrequent negative sentiment cases
- Generative AI integration is not yet production ready due to API cost and reliability constraints

### **Future Improvements**

- Enhance feature extraction through domain specific aspect vocabularies
- Introduce confidence thresholding to filter low quality predictions
- Incorporate temporal analysis to track sentiment changes over time, such as post product updates
- Deploy generative AI components for scalable natural language insights
- Implement a real time processing API for live review analysis
- Expand support for multilingual reviews in global markets
- Develop comparative sentiment analysis across competing products

---

## 📁 Repository Structure

```
cadence_2B_bttai/
├── dashboard/
│   └── dashboard.py          # Streamlit dashboard
│
├── notebooks/
│   ├── 00_Generate_Disagreements.ipynb    # Disagreement detection analysis
│   ├── 01_Data_Overview.ipynb             # Exploratory data analysis
│   ├── 02_Aspect_Analysis.ipynb           # Aspect extraction and analysis
│   ├── 03_Disagreement_Analysis.ipynb     # Rating-sentiment mismatch investigation
│   └── convert_reviews_to_parquet.ipynb   # Data preprocessing and format conversion
│
├── data/
│   ├── aspect_sentiments.csv          # ABSA model output
│   ├── aspect_summary.csv             # Aggregated statistics by aspect
│   ├── disagreements.csv              # Rating-sentiment mismatches
│   ├── disagreement_insights.json     # Disagreement pattern metadata
│   ├── ai_batch_categorized.csv       # Gemini batch processing results
│   └── sample_reviews.csv             # Sample dataset for testing
│
├── models/
│   └── absa.py                        # ABSA model implementation
│
├── Cadence 2B Project Presentation.pdf    # Final presentation slides
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── .gitignore                         # Git ignore configuration
```

---

## ~ **Acknowledgements** ~

Thank you Dr. Farhan Raseed, Matt Brems, and the entire Break Through Tech team for helping us throughout this process and giving us the opportunity to build this project!

Special thanks to:

- **Matt Brems** for technical guidance on model selection, debugging, and production best practices
- **Farhan Raseed** for keeping us focused on real-world business impact and providing significant educationr resources
- **Hugging Face community** for pre-trained models that accelerated development
- **Cadence Design Systems** for presenting a challenging real-world problem
