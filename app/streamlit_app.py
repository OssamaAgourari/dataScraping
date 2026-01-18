"""
AI Model Performance Analysis Dashboard
Main Streamlit application for visualizing and predicting AI model performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.data_viewer import render_data_viewer
from app.components.visualization import render_visualizations
from app.components.prediction import render_prediction_page

# Page configuration
st.set_page_config(
    page_title="AI Model Performance Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


def get_data_path():
    """Get path to data directory"""
    return Path(__file__).parent.parent / 'data'


@st.cache_data
def load_data():
    """Load and cache the cleaned dataset"""
    data_path = get_data_path() / 'processed' / 'cleaned_data.csv'

    if data_path.exists():
        df = pd.read_csv(data_path)
        return df
    else:
        st.warning("No cleaned data found. Please run the data pipeline first.")
        return None


@st.cache_resource
def load_model():
    """Load and cache the trained model"""
    models_path = get_data_path() / 'models'

    if models_path.exists():
        model_files = list(models_path.glob('*.joblib'))
        if model_files:
            model = joblib.load(model_files[0])
            return model

    return None


def render_home_page(df):
    """Render the home page with overview statistics"""

    st.markdown('<h1 class="main-header">AI Model Performance Analysis Dashboard</h1>', unsafe_allow_html=True)

    st.markdown("""
    Welcome to the AI Model Performance Analysis Dashboard. This tool provides insights into
    AI model benchmarks scraped from Papers with Code, including performance trends,
    architecture comparisons, and predictive modeling.
    """)

    st.divider()

    if df is None:
        st.error("No data available. Please run the data pipeline first.")
        st.code("""
# Run these commands to generate data:
python src/scraper.py
python src/cleaner.py
python src/models.py
        """)
        return

    # Key Metrics
    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Models",
            value=f"{len(df):,}",
            delta=None
        )

    with col2:
        st.metric(
            label="Unique Tasks",
            value=df['task'].nunique() if 'task' in df.columns else 0
        )

    with col3:
        avg_score = df['score'].mean() if 'score' in df.columns else 0
        st.metric(
            label="Average Score",
            value=f"{avg_score:.2f}"
        )

    with col4:
        max_score = df['score'].max() if 'score' in df.columns else 0
        st.metric(
            label="Best Score",
            value=f"{max_score:.2f}"
        )

    st.divider()

    # Quick Stats
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Models")
        if 'score' in df.columns and 'model_name' in df.columns:
            top_models = df.nlargest(10, 'score')[['model_name', 'score', 'task']].reset_index(drop=True)
            top_models.index = top_models.index + 1
            st.dataframe(top_models, use_container_width=True)

    with col2:
        st.subheader("Performance Distribution")
        if 'score' in df.columns:
            fig = px.histogram(
                df, x='score', nbins=30,
                title='Score Distribution',
                color_discrete_sequence=['#3498db']
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Task breakdown
    st.subheader("Models by Task")
    if 'task' in df.columns:
        task_counts = df['task'].value_counts().reset_index()
        task_counts.columns = ['Task', 'Count']

        fig = px.bar(
            task_counts.head(10),
            x='Task', y='Count',
            title='Number of Models per Task (Top 10)',
            color='Count',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Recent data info
    st.divider()
    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"**Columns:** {len(df.columns)}")
        st.write("Columns:", ", ".join(df.columns.tolist()[:10]) + ("..." if len(df.columns) > 10 else ""))

    with col2:
        if 'year' in df.columns:
            year_range = f"{int(df['year'].min())} - {int(df['year'].max())}"
            st.info(f"**Year Range:** {year_range}")

    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        st.info(f"**Data Completeness:** {100-missing_pct:.1f}%")


def render_insights_page(df):
    """Render the insights and conclusions page"""

    st.header("Insights & Conclusions")

    if df is None:
        st.error("No data available.")
        return

    st.subheader("Key Findings")

    # Best model
    if 'score' in df.columns and 'model_name' in df.columns:
        best_model = df.loc[df['score'].idxmax()]
        st.success(f"**Best Performing Model:** {best_model['model_name']} with score {best_model['score']:.2f}")

    # Architecture insights
    if 'architecture_family' in df.columns:
        best_arch = df.groupby('architecture_family')['score'].mean().idxmax()
        avg_score = df.groupby('architecture_family')['score'].mean().max()
        st.info(f"**Best Architecture Family:** {best_arch} (avg score: {avg_score:.2f})")

    # Year trends
    if 'year' in df.columns and 'score' in df.columns:
        yearly = df.groupby('year')['score'].mean()
        if len(yearly) >= 2:
            recent_trend = yearly.iloc[-1] - yearly.iloc[-2]
            if recent_trend > 0:
                st.info(f"**Trend:** Performance improved by {recent_trend:.2f} points in the most recent year")
            else:
                st.warning(f"**Trend:** Performance decreased by {abs(recent_trend):.2f} points in the most recent year")

    st.divider()

    st.subheader("Observations")

    observations = [
        "Transformer-based architectures dominate recent benchmarks across multiple tasks",
        "Larger models generally achieve better performance, but with diminishing returns",
        "The rate of improvement in benchmark scores has been accelerating in recent years",
        "Cross-task transfer learning is becoming increasingly important",
        "Efficiency metrics (parameters vs performance) are gaining attention"
    ]

    for obs in observations:
        st.write(f"- {obs}")

    st.divider()

    st.subheader("Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **For Researchers:**
        - Focus on efficiency improvements alongside raw performance
        - Consider multi-task and transfer learning approaches
        - Benchmark on diverse datasets to ensure generalization
        """)

    with col2:
        st.markdown("""
        **For Practitioners:**
        - Evaluate models on your specific use case, not just benchmark scores
        - Consider computational costs and deployment requirements
        - Stay updated on emerging architectures and techniques
        """)


def main():
    """Main application entry point"""

    # Sidebar navigation
    st.sidebar.title("Navigation")

    pages = {
        "Home": "home",
        "Data Explorer": "data",
        "Visualizations": "viz",
        "Make Predictions": "predict",
        "Insights": "insights"
    }

    selected_page = st.sidebar.radio("Go to", list(pages.keys()))

    # Load data
    df = load_data()

    # Load model
    model = load_model()

    # Sidebar info
    st.sidebar.divider()
    st.sidebar.subheader("About")
    st.sidebar.info("""
    This dashboard analyzes AI model performance data
    scraped from Papers with Code benchmarks.

    **Features:**
    - Explore benchmark data
    - Visualize trends
    - Predict model performance
    """)

    # Data status
    st.sidebar.divider()
    if df is not None:
        st.sidebar.success(f"Data loaded: {len(df)} records")
    else:
        st.sidebar.error("No data loaded")

    if model is not None:
        st.sidebar.success("Model loaded")
    else:
        st.sidebar.warning("No model loaded")

    # Render selected page
    if selected_page == "Home":
        render_home_page(df)
    elif selected_page == "Data Explorer":
        render_data_viewer(df)
    elif selected_page == "Visualizations":
        render_visualizations(df)
    elif selected_page == "Make Predictions":
        render_prediction_page(df, model)
    elif selected_page == "Insights":
        render_insights_page(df)


if __name__ == "__main__":
    main()
