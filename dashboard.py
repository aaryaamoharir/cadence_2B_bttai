import streamlit as st
import pandas as pd
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Review Insights Dashboard", layout="wide")

# Load data
try:
    data = pd.read_csv("dashboard_data.csv")
except FileNotFoundError:
    st.error("CSV file not found. Please generate 'dashboard_data.csv' in Colab first.")
    st.stop()

# Helper Function for Sentiment Category
def get_sentiment_category(score):
    if score < 0.4:
        return "Negative"
    elif score > 0.6:
        return "Positive"
    else:
        return "Neutral"

# Apply category for easier filtering/visuals later
data['sentiment_category'] = data['sentiment'].apply(get_sentiment_category)

# Sidebar filters
st.sidebar.title("Filters")

# Mentions Slider
max_mentions = int(data["mentions"].max()) if not data.empty else 10
min_mentions = st.sidebar.slider("Minimum Mentions", 0, max_mentions, 5)

# Sentiment Range Slider 
st.sidebar.subheader("Sentiment Filter")
sentiment_range = st.sidebar.slider(
    "Select Sentiment Range",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0), # Default to full range
    step=0.01,
    help="Filter features based on their average sentiment score."
)
min_sentiment, max_sentiment = sentiment_range

# Filter Data
filtered_data = data[
    (data["mentions"] >= min_mentions) & 
    (data["sentiment"] >= min_sentiment) &
    (data["sentiment"] <= max_sentiment)
]

# 3. Feature Search
if not filtered_data.empty:
    st.sidebar.subheader("Search")
    all_features = filtered_data["feature"].tolist()
    selected_features = st.sidebar.multiselect(
        "Find Specific Features", 
        all_features
    )
    if selected_features:
        filtered_data = filtered_data[filtered_data["feature"].isin(selected_features)]

# App Layout
st.title("Product Reviews Insights")
st.markdown("Analyze customer feedback patterns, sentiment, and feature popularity.")

# Metrics Overview
if not filtered_data.empty:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Features Found", f"{len(filtered_data)}")
    col2.metric("Total Mentions", f"{filtered_data['mentions'].sum():,}")
    
    avg_sent = filtered_data['sentiment'].mean()
    sent_color = "normal"
    if avg_sent > 0.6: sent_color = "normal" # Streamlit doesn't support green directly in metric delta color without delta
    
    col3.metric("Avg Sentiment", f"{avg_sent:.2f}")
    
    # Calculate most common sentiment category in filtered batch
    top_cat = filtered_data['sentiment_category'].mode()[0]
    col4.metric("Dominant Mood", top_cat)
else:
    st.warning("No data matches your filters. Adjust the sliders in the sidebar.")
    st.stop()

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Top Features", "Sentiment Analysis", "Word Cloud", "Raw Data"])

# Tab 1: Top Features
with tab1:
    st.subheader("Most Discussed Features")
    st.caption("Bars are colored by sentiment (Red=Neg, Yellow=Neu, Green=Pos)")
    
    # Prepare data: Sort by mentions
    chart_data = filtered_data.sort_values(by="mentions", ascending=False).head(20)
    
    # Altair Chart
    bar_chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('mentions', title='Number of Mentions'),
        y=alt.Y('feature', sort='-x', title='Feature'),
        color=alt.Color('sentiment', scale=alt.Scale(scheme='redyellowgreen', domain=[0, 1]), title='Sentiment'),
        tooltip=['feature', 'mentions', 'sentiment', 'sentiment_category']
    ).properties(
        height=500
    ).interactive()
    
    st.altair_chart(bar_chart, use_container_width=True)

# Tab 2: Sentiment Analysis
with tab2:
    st.subheader("Sentiment Distribution by Feature")
    
    # Sort by sentiment for clearer view
    sent_data = filtered_data.sort_values(by="sentiment", ascending=False).head(20)
    
    # Custom color scale for categories
    domain = ['Negative', 'Neutral', 'Positive']
    range_ = ['#FF4B4B', '#FFC107', '#28A745'] # Red, Amber, Green
    
    # Chart
    sent_chart = alt.Chart(sent_data).mark_bar().encode(
        x=alt.X('feature', sort='-y', title='Feature'),
        y=alt.Y('sentiment', title='Sentiment Score (0-1)'),
        color=alt.Color('sentiment_category', scale=alt.Scale(domain=domain, range=range_), title='Category'),
        tooltip=['feature', 'sentiment', 'mentions', 'sentiment_category']
    ).properties(
        height=500
    ).interactive()
    
    st.altair_chart(sent_chart, use_container_width=True)

# Tab 3: Word Cloud
with tab3:
    col_wc1, col_wc2 = st.columns([3, 1])
    with col_wc1:
        st.subheader("Feature Word Cloud")
        # Generate word cloud
        freq_dict = dict(zip(filtered_data["feature"], filtered_data["mentions"]))
        if freq_dict:
            wordcloud = WordCloud(
                width=800, height=400, 
                background_color="white", 
                colormap="viridis",
                max_words=100
            ).generate_from_frequencies(freq_dict)
            
            # Display using matplotlib to remove padding
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("Not enough data to generate a word cloud.")
    
    with col_wc2:
        st.markdown("### Top Keywords")
        st.dataframe(filtered_data[['feature', 'mentions']].sort_values('mentions', ascending=False).head(10), hide_index=True)

# Tab 4: Raw Data
with tab4:
    st.subheader("Data Explorer")
    st.dataframe(filtered_data, use_container_width=True)