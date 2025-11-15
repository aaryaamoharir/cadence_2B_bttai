import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --- Load data ---
data = pd.read_csv("abba_features.csv")

# --- Page config ---
st.set_page_config(page_title="Amazon Review Dashboard", layout="wide")

# --- Sidebar filters ---
st.sidebar.header("Filters")
min_mentions = st.sidebar.slider("Minimum Mentions", 0, int(data["mentions"].max()), 0)
min_sentiment = st.sidebar.slider("Minimum Sentiment", 0.0, 1.0, 0.0, 0.01)
selected_features = st.sidebar.multiselect("Select Features", data["feature"].tolist(), default=data["feature"].tolist())

# Filter data based on sidebar inputs
filtered_data = data[
    (data["mentions"] >= min_mentions) &
    (data["sentiment"] >= min_sentiment) &
    (data["feature"].isin(selected_features))
]

# --- App title ---
st.title("📊 Amazon Review Feature Dashboard")
st.write("Interactive dashboard showing ABBA AI model results.")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Top Features", "Sentiment Analysis", "Word Cloud", "Raw Data"])

# --- Tab 1: Top Features ---
with tab1:
    st.subheader("Top Mentioned Features")
    st.bar_chart(filtered_data.set_index("feature")["mentions"])

# --- Tab 2: Sentiment Analysis ---
with tab2:
    st.subheader("Sentiment by Feature")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(filtered_data["feature"], filtered_data["sentiment"], color="skyblue")
    ax.set_ylabel("Sentiment (0–1)")
    ax.set_xticks(range(len(filtered_data["feature"])))
    ax.set_xticklabels(filtered_data["feature"], rotation=45, ha="right")
    st.pyplot(fig)

# --- Tab 3: Word Cloud ---
with tab3:
    st.subheader("Feature Word Cloud")
    if not filtered_data.empty:
        wordcloud = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(
            dict(zip(filtered_data["feature"], filtered_data["mentions"]))
        )
        st.image(wordcloud.to_array())
    else:
        st.write("No data to display for word cloud with current filters.")

# --- Tab 4: Raw Data ---
with tab4:
    st.subheader("Raw Data Table")
    st.dataframe(filtered_data)
