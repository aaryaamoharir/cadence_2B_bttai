# Standard libraries
import json       # Parse JSON configuration files
import numpy as np    # Numerical operations

# Third party libraries
import streamlit as st         # Generate interactive dashboard
import pandas as pd             # Data manipulation and analysis
import plotly.express as px     # Plotting interface
import plotly.graph_objects as go    # Plotting for custom charts
from plotly.subplots import make_subplots    # Create multi-panel figures

# Page configuration must be first Streamlit command
st.set_page_config(
    page_title="Amazon Review Intelligence | Cadence 2B",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Alert box styling */
    .insight-box {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .danger-box {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Section divider styling */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Review card styling */
    .review-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .review-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Data loading function with caching
@st.cache_data
def load_data():
    """
    Load all CSV and JSON data files needed for dashboard
    Returns tuple of dataframes and success boolean
    """
    try:
        # Load ABSA model output with aspect level sentiment results
        aspect_sentiments = pd.read_csv("data/aspect_sentiments.csv")
        
        # Load aggregated statistics by aspect and sentiment
        aspect_summary = pd.read_csv("data/aspect_summary.csv")
        
        # Load cases where star rating contradicts text sentiment
        disagreements = pd.read_csv("data/disagreements.csv")
        
        # Load metadata about disagreement patterns
        with open("data/disagreement_insights.json", 'r') as f:
            disagreement_insights = json.load(f)
        
        return aspect_sentiments, aspect_summary, disagreements, disagreement_insights, True
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure all data files are in the same directory as this dashboard.")
        return None, None, None, None, False

# Load all data
aspect_sentiments, aspect_summary, disagreements, disagreement_insights, data_loaded = load_data()

# Stop execution if data loading failed
if not data_loaded:
    st.stop()

# Header section
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="main-title">Amazon Review Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Aspect-Based Sentiment Analysis | Cadence 2B Fall 2025</p>', unsafe_allow_html=True)

with col2:
    st.image("dashboard/cadence_logo.png")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Sidebar filters
st.sidebar.title("Filter Controls")
st.sidebar.markdown("---")

# Sentiment filter
st.sidebar.subheader("Sentiment")
sentiment_options = ['All'] + list(aspect_sentiments['sentiment'].unique())
selected_sentiment = st.sidebar.multiselect(
    "Select Sentiments",
    sentiment_options,
    default=['All']
)

# Rating filter
st.sidebar.subheader("Star Rating")
min_rating = float(aspect_sentiments['rating'].min())
max_rating = float(aspect_sentiments['rating'].max())
rating_range = st.sidebar.slider(
    "Rating Range",
    min_value=min_rating,
    max_value=max_rating,
    value=(min_rating, max_rating),
    step=0.5
)

# Sentiment score filter
st.sidebar.subheader("Sentiment Score")
min_sent_score = st.sidebar.slider(
    "Minimum Sentiment Score",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    help="Filter by sentiment strength"
)

# Display options
st.sidebar.subheader("Display Options")
top_n = st.sidebar.slider(
    "Top Aspects to Show",
    min_value=5,
    max_value=30,
    value=20,
    step=5
)

st.sidebar.markdown("---")
st.sidebar.info("Tip: Adjust filters to explore specific product aspects and sentiment patterns.")

# Apply filters to create filtered dataframe
filtered_df = aspect_sentiments.copy()

# Apply sentiment filter if not 'all' selected
if 'All' not in selected_sentiment:
    filtered_df = filtered_df[filtered_df['sentiment'].isin(selected_sentiment)]

# Apply rating range filter
filtered_df = filtered_df[
    (filtered_df['rating'] >= rating_range[0]) &
    (filtered_df['rating'] <= rating_range[1])
]

# Apply sentiment score threshold filter
filtered_df = filtered_df[filtered_df['sentiment_score'] >= min_sent_score]

# Key metrics section
st.subheader("Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

# Total reviews metric
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Reviews</div>
        <div class="metric-value">{len(filtered_df):,}</div>
    </div>
    """, unsafe_allow_html=True)

# Positive percentage metric
with col2:
    positive_pct = (filtered_df['sentiment'].str.lower() == 'positive').sum() / len(filtered_df) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Positive %</div>
        <div class="metric-value">{positive_pct:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# Average sentiment score metric
with col3:
    avg_sent_score = filtered_df['sentiment_score'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Sentiment</div>
        <div class="metric-value">{avg_sent_score:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

# Unique aspects metric
with col4:
    unique_aspects = filtered_df['aspect'].nunique()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Aspects Found</div>
        <div class="metric-value">{unique_aspects:,}</div>
    </div>
    """, unsafe_allow_html=True)

# Average rating metric
with col5:
    avg_rating = filtered_df['rating'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Rating</div>
        <div class="metric-value">{avg_rating:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Disagreement alert box
if len(disagreements) > 0:
    st.markdown(f"""
    <div class="warning-box">
        <h3>Rating-Sentiment Disagreements Detected</h3>
        <p><strong>{len(disagreements):,} reviews</strong> show misalignment between star ratings and sentiment.</p>
        <p>These reviews may indicate:</p>
        <ul>
            <li>Complex customer opinions (loved some features, hated others)</li>
            <li>Misleading ratings (5 stars but negative text)</li>
            <li>Opportunity areas for product improvement</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main analysis tabs
tab1, tab2, tab3 = st.tabs([
    "Aspect Overview",
    "Top Insights",
    "Disagreements"
])

# Tab 1 - Aspect Overview
with tab1:
    st.subheader("Top Aspects by Mention Count")
    
    # Calculate how many times each aspect appears
    aspect_counts = filtered_df['aspect'].value_counts().head(top_n).reset_index()
    aspect_counts.columns = ['aspect', 'count']
    



    # Add sentiment breakdown for each aspect
    temp_df = filtered_df.copy()
    temp_df['sentiment'] = temp_df['sentiment'].str.capitalize()    # Capitalize for chart
    aspect_sentiment_breakdown = temp_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
    aspect_counts = aspect_counts.merge(
        aspect_sentiment_breakdown,
        left_on='aspect',
        right_index=True,
        how='left'
    )
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Create stacked horizontal bar chart
        fig = go.Figure()
        
        # Add positive sentiment bars
        if 'Positive' in aspect_counts.columns:
            fig.add_trace(go.Bar(
                name='Positive',
                y=aspect_counts['aspect'],
                x=aspect_counts['Positive'],
                orientation='h',
                marker_color='#10b981',
                text=aspect_counts['Positive'],
                textposition='inside'
            ))
        
        # Add neutral sentiment bars
        if 'Neutral' in aspect_counts.columns:
            fig.add_trace(go.Bar(
                name='Neutral',
                y=aspect_counts['aspect'],
                x=aspect_counts['Neutral'],
                orientation='h',
                marker_color='#6b7280',
                text=aspect_counts['Neutral'],
                textposition='inside'
            ))
        
        # Add negative sentiment bars
        if 'Negative' in aspect_counts.columns:
            fig.add_trace(go.Bar(
                name='Negative',
                y=aspect_counts['aspect'],
                x=aspect_counts['Negative'],
                orientation='h',
                marker_color='#ef4444',
                text=aspect_counts['Negative'],
                textposition='inside'
            ))
        
        # Update chart layout
        fig.update_layout(
            title=f"Top {top_n} Product Aspects - Sentiment Breakdown",
            barmode='stack',
            height=600,
            xaxis_title="Number of Mentions",
            yaxis_title="",
            yaxis={'categoryorder': 'total ascending'},
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Sentiment Distribution")
        
        # Create overall sentiment pie chart
        sentiment_counts = filtered_df['sentiment'].value_counts()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.4,
            marker_colors=['#10b981', '#6b7280', '#ef4444'],
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig_pie.update_layout(
            height=300,
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("### Quick Stats")
        st.metric("Most Mentioned", aspect_counts.iloc[0]['aspect'], 
                 f"{aspect_counts.iloc[0]['count']:,} mentions")
        
        st.metric("Highest Rated Aspect", 
                 filtered_df.groupby('aspect')['rating'].mean().idxmax(),
                 f"{filtered_df.groupby('aspect')['rating'].mean().max():.2f}")

# Tab 2 - Top Insights
with tab2:
    st.subheader("Actionable Product Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top Strengths")
        st.markdown("*Aspects with highest positive sentiment*")
        
        # Calculate aspects with high positive sentiment
        positive_aspects = filtered_df[filtered_df['sentiment'].str.lower() == 'positive']
        top_positive = positive_aspects.groupby('aspect').agg({
            'sentiment_score': 'mean',
            'aspect': 'count'
        }).rename(columns={'aspect': 'count'})
        
        # Filter for aspects with at least 10 mentions
        top_positive = top_positive[top_positive['count'] >= 10].sort_values('sentiment_score', ascending=False).head(10)
        
        # Display top positive aspects
        for idx, (aspect, row) in enumerate(top_positive.iterrows(), 1):
            st.markdown(f"""
            <div class="review-card">
                <strong style="color: #059669; font-size: 1.1rem;">#{idx} {aspect.title()}</strong><br/>
                <span style="color: #6b7280;">Avg Sentiment Score: <strong>{row['sentiment_score']:.3f}</strong></span><br/>
                <span style="color: #6b7280;">Positive Mentions: <strong>{int(row['count'])}</strong></span>
            </div>
            """, unsafe_allow_html=True)
    


    with col2:
        st.markdown("### Top Pain Points")
        st.markdown("*Aspects with highest negative sentiment*")
        
        # Calculate aspects with high negative sentiment
        negative_aspects = filtered_df[filtered_df['sentiment'].str.lower() == 'negative']
        top_negative = negative_aspects.groupby('aspect').agg({
            'sentiment_score': 'mean',
            'aspect': 'count'
        }).rename(columns={'aspect': 'count'})
        
        # Filter for aspects with at least 10 mentions
        top_negative = top_negative[top_negative['count'] >= 10].sort_values('sentiment_score', ascending=True).head(10)
        
        # Display top negative aspects
        for idx, (aspect, row) in enumerate(top_negative.iterrows(), 1):
            st.markdown(f"""
            <div class="review-card">
                <strong style="color: #dc2626; font-size: 1.1rem;">#{idx} {aspect.title()}</strong><br/>
                <span style="color: #6b7280;">Avg Sentiment Score: <strong>{row['sentiment_score']:.3f}</strong></span><br/>
                <span style="color: #6b7280;">Negative Mentions: <strong>{int(row['count'])}</strong></span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Sentiment score distribution
    st.markdown("### Sentiment Score Distribution")
    
    fig = go.Figure(data=[go.Histogram(
        x=filtered_df['sentiment_score'],
        nbinsx=30,
        marker_color='#667eea'
    )])
    
    fig.update_layout(
        xaxis_title="Sentiment Score",
        yaxis_title="Count",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 3 - Disagreements Analysis
with tab3:
    st.subheader("Rating-Sentiment Disagreement Analysis")
    
    # Explanation of disagreements
    st.markdown("""
    <div class="insight-box">
        <h4>What are disagreements?</h4>
        <p>These are reviews where the star rating doesn't match the sentiment of the text:</p>
        <ul>
            <li><strong>High rating + Negative sentiment</strong>: Customer gave 4-5 stars but the review text is negative</li>
            <li><strong>Low rating + Positive sentiment</strong>: Customer gave 1-2 stars but the review text is positive</li>
        </ul>
        <p>These cases reveal complex customer opinions and potential areas for deeper investigation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(disagreements) > 0:
        # Disagreement breakdown
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### Found {len(disagreements):,} Disagreements")
            
            # Categorize disagreement types
            if 'rating' in disagreements.columns and 'sentiment' in disagreements.columns:
                disagreements['type'] = disagreements.apply(
                    lambda x: 'High Rating, Negative Text' if x['rating'] >= 4 and x['sentiment'].lower() == 'negative'
                    else 'Low Rating, Positive Text' if x['rating'] <= 2 and x['sentiment'].lower() == 'positive'
                    else 'Other Mismatch',
                    axis=1
                )
                
                type_counts = disagreements['type'].value_counts()
                
                # Create bar chart of disagreement types
                fig = go.Figure(data=[go.Bar(
                    x=type_counts.index,
                    y=type_counts.values,
                    marker_color=['#ef4444', '#f59e0b', '#6b7280'],
                    text=type_counts.values,
                    textposition='outside'
                )])
                
                fig.update_layout(
                    title="Disagreement Types",
                    xaxis_title="Type",
                    yaxis_title="Count",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Most Common Aspects in Disagreements")
            
            # Show aspects that appear most in disagreements
            if 'aspect' in disagreements.columns:
                top_disagreement_aspects = disagreements['aspect'].value_counts().head(10)
                
                for aspect, count in top_disagreement_aspects.items():
                    st.markdown(f"""
                    <div style="background: #f9fafb; padding: 0.75rem; margin: 0.5rem 0; border-radius: 6px;">
                        <strong>{aspect}</strong><br/>
                        <span style="color: #6b7280; font-size: 0.9rem;">{count} disagreements</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Sample disagreement cases
        st.markdown("### Sample Disagreement Cases")
        st.markdown("*Examples of reviews where rating and sentiment don't align*")
        
        sample_disagreements = disagreements.head(10)
        
        # Display each disagreement case
        for idx, row in sample_disagreements.iterrows():
            rating_stars = '★' * int(row.get('rating', 0))
            sentiment_color = '#10b981' if row.get('sentiment', '').lower() == 'positive' else '#ef4444'
            
            st.markdown(f"""
            <div class="review-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span><strong>Aspect:</strong> {row.get('aspect', 'N/A')}</span>
                    <span>{rating_stars}</span>
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <span style="background: {sentiment_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.85rem;">
                        {row.get('sentiment', 'N/A')}
                    </span>
                    <span style="margin-left: 1rem; color: #6b7280;">
                        Score: {row.get('sentiment_score', 0):.2f}
                    </span>
                </div>
                <p style="color: #374151; font-style: italic; margin: 0;">
                    "{row.get('text', 'No text available')[:200]}..."
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No disagreements found with current filters.")




# Footer
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem 0;'>
    <p style='font-size: 0.9rem; margin-bottom: 0.5rem;'>
        <strong>Amazon Review Intelligence Dashboard</strong> | Aspect-Based Sentiment Analysis
    </p>
    <p style='font-size: 0.8rem; margin: 0;'>
        Cadence 2B | Fall 2025 | Built with Streamlit & Plotly
    </p>
    <p style='font-size: 0.75rem; margin-top: 1rem; color: #9ca3af;'>
        Powered by DeBERTa v3 ABSA Model | 88% Sentiment Accuracy
    </p>
</div>
""", unsafe_allow_html=True)