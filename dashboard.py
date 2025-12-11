import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Product Reviews Dashboard", layout="wide")

# loading the csv 
@st.cache_data
def load_data():
    df = pd.read_csv("aspect_sentiments.csv")
    return df

df = load_data()

st.title("Product Reviews Dashboard")


has_rating = 'rating' in df.columns

# Wanted to add in sidebar features so that it's easier to filter 
st.sidebar.header("Filters")

sentiment_filter = st.sidebar.multiselect(
    "Filter by sentiment",
    options=["Positive", "Negative", "Neutral"],
    default=["Positive", "Negative", "Neutral"]
)
# Apply sentiment filter
filtered_df = df[df['sentiment'].str.capitalize().isin(sentiment_filter)]


if has_rating:
    min_rating, max_rating = st.sidebar.slider(
        "Filter by rating",
        min_value=float(df['rating'].min()),
        max_value=float(df['rating'].max()),
        value=(float(df['rating'].min()), float(df['rating'].max()))
    )
    filtered_df = filtered_df[
        (filtered_df['rating'] >= min_rating) &
        (filtered_df['rating'] <= max_rating)
    ]

# highlights the top metrics 
if has_rating:
    col1, col2, col3, col4 = st.columns(4)
else:
    col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Reviews", len(filtered_df))
    
with col2:
    positive_pct = (filtered_df['sentiment'].str.lower() == 'positive').sum() / len(filtered_df) * 100
    st.metric("Positive Reviews", f"{positive_pct:.1f}%")
    
with col3:
    negative_pct = (filtered_df['sentiment'].str.lower() == 'negative').sum() / len(filtered_df) * 100
    st.metric("Negative Reviews", f"{negative_pct:.1f}%")

if has_rating:    
    with col4:
        st.metric("Avg Rating", f"{filtered_df['rating'].mean():.2f}")

st.divider()

# handles the sentiment distribution 
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = filtered_df['sentiment'].str.capitalize().value_counts()
    fig_pie = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        color=sentiment_counts.index,
        color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("⭐ Rating Distribution")
    if has_rating:
        rating_counts = filtered_df['rating'].value_counts().sort_index()
        fig_bar = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            labels={'x': 'Rating', 'y': 'Count'},
            color=rating_counts.values,
            color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Rating column not available in dataset")

st.divider()

# analyzes top features 
st.subheader("🏆 Top 20 Features by Sentiment")

tab1, tab2, tab3 = st.tabs(["All Sentiments", "Positive Features", "Negative Features"])

with tab1:
    # top 20 features by absolute sentiment score
    top_features = filtered_df.groupby('aspect').agg({
        'sentiment_score': 'mean',
        'review_id': 'count'
    }).rename(columns={'review_id': 'mention_count'})
    
    top_features['abs_score'] = abs(top_features['sentiment_score'])
    top_features = top_features.sort_values('abs_score', ascending=False).head(20)
    
    fig = px.bar(
        top_features.reset_index(),
        x='sentiment_score',
        y='aspect',
        orientation='h',
        color='sentiment_score',
        color_continuous_scale=['#e74c3c', '#f39c12', '#2ecc71'],
        labels={'sentiment_score': 'Avg Sentiment Score', 'aspect': 'Feature'},
        text='mention_count'
    )
    fig.update_traces(texttemplate='%{text} mentions', textposition='outside')
    fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # top 20 positive features 
    positive_df = filtered_df[filtered_df['sentiment'].str.lower() == 'positive']
    top_positive = positive_df.groupby('aspect').agg({
        'sentiment_score': 'mean',
        'review_id': 'count'
    }).rename(columns={'review_id': 'mention_count'})
    
    top_positive = top_positive.sort_values('sentiment_score', ascending=False).head(20)
    
    fig_pos = px.bar(
        top_positive.reset_index(),
        x='sentiment_score',
        y='aspect',
        orientation='h',
        color='sentiment_score',
        color_continuous_scale='Greens',
        labels={'sentiment_score': 'Avg Sentiment Score', 'aspect': 'Feature'},
        text='mention_count'
    )
    fig_pos.update_traces(texttemplate='%{text} mentions', textposition='outside')
    fig_pos.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_pos, use_container_width=True)
    
    st.dataframe(
        top_positive.reset_index().rename(columns={
            'aspect': 'Feature',
            'sentiment_score': 'Avg Score',
            'mention_count': 'Mentions'
        }),
        use_container_width=True,
        hide_index=True
    )

with tab3:
    # Top 20 negative features
    negative_df = filtered_df[filtered_df['sentiment'].str.lower() == 'negative']
    top_negative = negative_df.groupby('aspect').agg({
        'sentiment_score': 'mean',
        'review_id': 'count'
    }).rename(columns={'review_id': 'mention_count'})
    
    top_negative = top_negative.sort_values('sentiment_score', ascending=True).head(20)
    
    fig_neg = px.bar(
        top_negative.reset_index(),
        x='sentiment_score',
        y='aspect',
        orientation='h',
        color='sentiment_score',
        color_continuous_scale='Reds',
        labels={'sentiment_score': 'Avg Sentiment Score', 'aspect': 'Feature'},
        text='mention_count'
    )
    fig_neg.update_traces(texttemplate='%{text} mentions', textposition='outside')
    fig_neg.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_neg, use_container_width=True)
    
    st.dataframe(
        top_negative.reset_index().rename(columns={
            'aspect': 'Feature',
            'sentiment_score': 'Avg Score',
            'mention_count': 'Mentions'
        }),
        use_container_width=True,
        hide_index=True
    )

st.divider()

# Feature sentiment breakdown
st.subheader("🔍 Feature Sentiment Breakdown")

# positive/negative/neutral by feature
feature_sentiment = pd.crosstab(
    filtered_df['aspect'],
    filtered_df['sentiment'].str.capitalize()
).reset_index()

# top 15 
top_features_list = filtered_df['aspect'].value_counts().head(15).index.tolist()
feature_sentiment_top = feature_sentiment[feature_sentiment['aspect'].isin(top_features_list)]

fig_stacked = go.Figure()

if 'Positive' in feature_sentiment_top.columns:
    fig_stacked.add_trace(go.Bar(
        y=feature_sentiment_top['aspect'],
        x=feature_sentiment_top['Positive'],
        name='Positive',
        orientation='h',
        marker_color='#2ecc71'
    ))

if 'Neutral' in feature_sentiment_top.columns:
    fig_stacked.add_trace(go.Bar(
        y=feature_sentiment_top['aspect'],
        x=feature_sentiment_top['Neutral'],
        name='Neutral',
        orientation='h',
        marker_color='#95a5a6'
    ))

if 'Negative' in feature_sentiment_top.columns:
    fig_stacked.add_trace(go.Bar(
        y=feature_sentiment_top['aspect'],
        x=feature_sentiment_top['Negative'],
        name='Negative',
        orientation='h',
        marker_color='#e74c3c'
    ))

fig_stacked.update_layout(
    barmode='stack',
    height=500,
    xaxis_title='Number of Reviews',
    yaxis_title='Feature',
    legend_title='Sentiment'
)

st.plotly_chart(fig_stacked, use_container_width=True)

st.divider()

# Raw data section
with st.expander("📋 View Raw Data"):
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="filtered_reviews.csv",
        mime="text/csv"
    )

# summary stats 
with st.expander("📈 Summary Statistics"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sentiment Statistics**")
        st.write(f"- Total Positive: {(filtered_df['sentiment'].str.lower() == 'positive').sum()}")
        st.write(f"- Total Negative: {(filtered_df['sentiment'].str.lower() == 'negative').sum()}")
        st.write(f"- Total Neutral: {(filtered_df['sentiment'].str.lower() == 'neutral').sum()}")
        st.write(f"- Avg Sentiment Score: {filtered_df['sentiment_score'].mean():.3f}")
    
    with col2:
        st.write("**Feature Statistics**")
        st.write(f"- Total Unique Features: {filtered_df['aspect'].nunique()}")
        st.write(f"- Most Mentioned Feature: {filtered_df['aspect'].mode()[0] if len(filtered_df) > 0 else 'N/A'}")
        st.write(f"- Avg Mentions per Feature: {len(filtered_df) / filtered_df['aspect'].nunique():.1f}")