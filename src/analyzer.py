"""
Exploratory Data Analysis Module for AI Model Performance Analysis
Generates statistics, visualizations, and insights from the cleaned data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_cleaned_data(filepath=None):
    """
    Load cleaned data for analysis

    Parameters:
    -----------
    filepath : str or Path, optional
        Path to cleaned data CSV

    Returns:
    --------
    pandas.DataFrame : Cleaned data
    """
    if filepath is None:
        filepath = Path(__file__).parent.parent / 'data' / 'processed' / 'cleaned_data.csv'

    logger.info(f"Loading cleaned data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} records")

    return df


def generate_summary_stats(df):
    """
    Generate comprehensive summary statistics

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe

    Returns:
    --------
    dict : Dictionary containing various statistics
    """
    logger.info("Generating summary statistics...")

    stats = {
        'total_models': len(df),
        'unique_models': df['model_name'].nunique() if 'model_name' in df.columns else 0,
        'unique_tasks': df['task'].nunique() if 'task' in df.columns else 0,
        'unique_datasets': df['dataset'].nunique() if 'dataset' in df.columns else 0,
    }

    # Score statistics
    if 'score' in df.columns:
        score_stats = df['score'].describe()
        stats['score_stats'] = {
            'mean': round(score_stats['mean'], 2),
            'median': round(score_stats['50%'], 2),
            'std': round(score_stats['std'], 2),
            'min': round(score_stats['min'], 2),
            'max': round(score_stats['max'], 2),
        }

    # Year distribution
    if 'year' in df.columns:
        year_counts = df['year'].value_counts().sort_index()
        stats['year_distribution'] = year_counts.to_dict()

    # Task distribution
    if 'task' in df.columns:
        stats['task_distribution'] = df['task'].value_counts().to_dict()

    # Architecture distribution
    if 'architecture_family' in df.columns:
        stats['architecture_distribution'] = df['architecture_family'].value_counts().to_dict()

    # Model size distribution
    if 'model_size_category' in df.columns:
        stats['size_distribution'] = df['model_size_category'].value_counts().to_dict()

    return stats


def plot_performance_distribution(df, save_path=None):
    """
    Plot the distribution of performance scores

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    save_path : str or Path, optional
        Path to save the figure

    Returns:
    --------
    plotly.graph_objects.Figure : The plotly figure
    """
    logger.info("Creating performance distribution plot...")

    if 'score' not in df.columns:
        logger.warning("No 'score' column found")
        return None

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Score Distribution (Histogram)', 'Score Distribution (Box Plot)'))

    # Histogram
    fig.add_trace(
        go.Histogram(x=df['score'], nbinsx=30, name='Score Distribution',
                     marker_color='#3498db', opacity=0.7),
        row=1, col=1
    )

    # Box plot
    fig.add_trace(
        go.Box(y=df['score'], name='Scores', marker_color='#2ecc71'),
        row=1, col=2
    )

    fig.update_layout(
        title='AI Model Performance Score Distribution',
        showlegend=False,
        height=400
    )

    fig.update_xaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=2)

    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_top_models(df, n=15, save_path=None):
    """
    Plot top performing models

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    n : int
        Number of top models to show
    save_path : str or Path, optional
        Path to save the figure

    Returns:
    --------
    plotly.graph_objects.Figure : The plotly figure
    """
    logger.info(f"Creating top {n} models plot...")

    if 'score' not in df.columns or 'model_name' not in df.columns:
        logger.warning("Required columns not found")
        return None

    # Get top models by score
    top_models = df.nlargest(n, 'score')[['model_name', 'score', 'task', 'dataset']].copy()

    # Create color based on task
    colors = px.colors.qualitative.Set3

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=top_models['model_name'],
        x=top_models['score'],
        orientation='h',
        marker_color=colors[:len(top_models)],
        text=top_models['score'].round(2),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Score: %{x:.2f}<br>Task: %{customdata[0]}<br>Dataset: %{customdata[1]}<extra></extra>',
        customdata=top_models[['task', 'dataset']].values
    ))

    fig.update_layout(
        title=f'Top {n} Performing AI Models',
        xaxis_title='Performance Score',
        yaxis_title='Model',
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=150)
    )

    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_performance_trends(df, save_path=None):
    """
    Plot performance trends over time

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    save_path : str or Path, optional
        Path to save the figure

    Returns:
    --------
    plotly.graph_objects.Figure : The plotly figure
    """
    logger.info("Creating performance trends plot...")

    if 'year' not in df.columns or 'score' not in df.columns:
        logger.warning("Required columns not found")
        return None

    # Calculate yearly statistics
    yearly_stats = df.groupby('year').agg({
        'score': ['mean', 'max', 'min', 'count']
    }).round(2)
    yearly_stats.columns = ['mean_score', 'max_score', 'min_score', 'count']
    yearly_stats = yearly_stats.reset_index()

    fig = go.Figure()

    # Add mean score line
    fig.add_trace(go.Scatter(
        x=yearly_stats['year'],
        y=yearly_stats['mean_score'],
        mode='lines+markers',
        name='Average Score',
        line=dict(color='#3498db', width=3),
        marker=dict(size=10)
    ))

    # Add max score line
    fig.add_trace(go.Scatter(
        x=yearly_stats['year'],
        y=yearly_stats['max_score'],
        mode='lines+markers',
        name='Best Score',
        line=dict(color='#2ecc71', width=2, dash='dash'),
        marker=dict(size=8)
    ))

    # Add confidence band (min to max)
    fig.add_trace(go.Scatter(
        x=list(yearly_stats['year']) + list(yearly_stats['year'][::-1]),
        y=list(yearly_stats['max_score']) + list(yearly_stats['min_score'][::-1]),
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Score Range',
        hoverinfo='skip'
    ))

    fig.update_layout(
        title='AI Model Performance Trends Over Time',
        xaxis_title='Year',
        yaxis_title='Performance Score',
        height=450,
        hovermode='x unified'
    )

    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_task_comparison(df, save_path=None):
    """
    Compare performance across different tasks

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    save_path : str or Path, optional
        Path to save the figure

    Returns:
    --------
    plotly.graph_objects.Figure : The plotly figure
    """
    logger.info("Creating task comparison plot...")

    if 'task' not in df.columns or 'score' not in df.columns:
        logger.warning("Required columns not found")
        return None

    # Calculate task statistics
    task_stats = df.groupby('task')['score'].agg(['mean', 'std', 'count']).reset_index()
    task_stats.columns = ['task', 'mean_score', 'std_score', 'count']
    task_stats = task_stats.sort_values('mean_score', ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=task_stats['task'],
        x=task_stats['mean_score'],
        orientation='h',
        marker_color=px.colors.qualitative.Pastel,
        error_x=dict(type='data', array=task_stats['std_score'], visible=True),
        text=task_stats['mean_score'].round(2),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Mean: %{x:.2f}<br>Std: %{error_x.array:.2f}<br>Models: %{customdata}<extra></extra>',
        customdata=task_stats['count']
    ))

    fig.update_layout(
        title='Performance Comparison Across Tasks',
        xaxis_title='Mean Performance Score',
        yaxis_title='Task',
        height=500,
        margin=dict(l=200)
    )

    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_architecture_analysis(df, save_path=None):
    """
    Analyze performance by architecture type

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    save_path : str or Path, optional
        Path to save the figure

    Returns:
    --------
    plotly.graph_objects.Figure : The plotly figure
    """
    logger.info("Creating architecture analysis plot...")

    arch_col = 'architecture_family' if 'architecture_family' in df.columns else 'architecture'

    if arch_col not in df.columns or 'score' not in df.columns:
        logger.warning("Required columns not found")
        return None

    fig = px.box(
        df,
        x=arch_col,
        y='score',
        color=arch_col,
        title='Performance Distribution by Architecture Type',
        labels={arch_col: 'Architecture', 'score': 'Performance Score'}
    )

    fig.update_layout(
        height=450,
        showlegend=False,
        xaxis_tickangle=-45
    )

    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_parameters_vs_performance(df, save_path=None):
    """
    Plot relationship between model size and performance

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    save_path : str or Path, optional
        Path to save the figure

    Returns:
    --------
    plotly.graph_objects.Figure : The plotly figure
    """
    logger.info("Creating parameters vs performance plot...")

    if 'parameters_billions' not in df.columns or 'score' not in df.columns:
        logger.warning("Required columns not found")
        return None

    # Filter out missing values
    plot_df = df.dropna(subset=['parameters_billions', 'score'])

    if len(plot_df) == 0:
        logger.warning("No data with both parameters and scores")
        return None

    fig = px.scatter(
        plot_df,
        x='parameters_billions',
        y='score',
        color='task' if 'task' in plot_df.columns else None,
        size='score',
        hover_name='model_name',
        title='Model Size vs Performance',
        labels={'parameters_billions': 'Parameters (Billions)', 'score': 'Performance Score'},
        log_x=True
    )

    fig.update_layout(
        height=500
    )

    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_correlation_matrix(df, save_path=None):
    """
    Plot correlation matrix for numeric features

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    save_path : str or Path, optional
        Path to save the figure

    Returns:
    --------
    plotly.graph_objects.Figure : The plotly figure
    """
    logger.info("Creating correlation matrix plot...")

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        logger.warning("Not enough numeric columns for correlation")
        return None

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title='Feature Correlation Matrix',
        aspect='auto'
    )

    fig.update_layout(
        height=500,
        width=600
    )

    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_yearly_model_count(df, save_path=None):
    """
    Plot number of models published per year

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    save_path : str or Path, optional
        Path to save the figure

    Returns:
    --------
    plotly.graph_objects.Figure : The plotly figure
    """
    logger.info("Creating yearly model count plot...")

    if 'year' not in df.columns:
        logger.warning("No 'year' column found")
        return None

    year_counts = df['year'].value_counts().sort_index().reset_index()
    year_counts.columns = ['year', 'count']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=year_counts['year'],
        y=year_counts['count'],
        marker_color='#9b59b6',
        text=year_counts['count'],
        textposition='outside'
    ))

    fig.update_layout(
        title='Number of AI Models Published Per Year',
        xaxis_title='Year',
        yaxis_title='Number of Models',
        height=400
    )

    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved plot to {save_path}")

    return fig


def generate_insights_report(df):
    """
    Generate automated insights from the data

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe

    Returns:
    --------
    dict : Dictionary of insights
    """
    logger.info("Generating insights report...")

    insights = {
        'key_findings': [],
        'trends': [],
        'recommendations': []
    }

    # Best performing model
    if 'score' in df.columns and 'model_name' in df.columns:
        best_model = df.loc[df['score'].idxmax()]
        insights['key_findings'].append(
            f"Best performing model: {best_model['model_name']} with score {best_model['score']:.2f}"
        )

    # Most common architecture
    if 'architecture_family' in df.columns:
        most_common_arch = df['architecture_family'].value_counts().idxmax()
        insights['key_findings'].append(
            f"Most common architecture family: {most_common_arch}"
        )

    # Performance trend
    if 'year' in df.columns and 'score' in df.columns:
        yearly_mean = df.groupby('year')['score'].mean()
        if len(yearly_mean) >= 2:
            recent_years = yearly_mean.tail(2)
            if recent_years.iloc[-1] > recent_years.iloc[-2]:
                insights['trends'].append("Performance is improving over recent years")
            else:
                insights['trends'].append("Performance plateau observed in recent years")

    # Best task category
    if 'task_category' in df.columns and 'score' in df.columns:
        best_category = df.groupby('task_category')['score'].mean().idxmax()
        insights['key_findings'].append(
            f"Highest average performance in: {best_category}"
        )

    # Model size insights
    if 'parameters_billions' in df.columns and 'score' in df.columns:
        corr = df[['parameters_billions', 'score']].corr().iloc[0, 1]
        if corr > 0.3:
            insights['trends'].append("Larger models tend to perform better")
        elif corr < -0.3:
            insights['trends'].append("Smaller models are competitive with larger ones")
        else:
            insights['trends'].append("Model size has limited correlation with performance")

    # Recommendations
    insights['recommendations'].append("Consider Transformer-based architectures for NLP tasks")
    insights['recommendations'].append("Monitor emerging model architectures for potential improvements")
    insights['recommendations'].append("Balance model size with computational requirements")

    return insights


def create_all_visualizations(df, output_dir=None):
    """
    Create all visualizations and save to output directory

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    output_dir : str or Path, optional
        Directory to save visualizations

    Returns:
    --------
    dict : Dictionary of figure objects
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'data' / 'visualizations'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}

    # Generate all plots
    figures['performance_distribution'] = plot_performance_distribution(
        df, output_dir / 'performance_distribution.html')

    figures['top_models'] = plot_top_models(
        df, n=15, save_path=output_dir / 'top_models.html')

    figures['performance_trends'] = plot_performance_trends(
        df, save_path=output_dir / 'performance_trends.html')

    figures['task_comparison'] = plot_task_comparison(
        df, save_path=output_dir / 'task_comparison.html')

    figures['architecture_analysis'] = plot_architecture_analysis(
        df, save_path=output_dir / 'architecture_analysis.html')

    figures['parameters_vs_performance'] = plot_parameters_vs_performance(
        df, save_path=output_dir / 'parameters_vs_performance.html')

    figures['correlation_matrix'] = plot_correlation_matrix(
        df, save_path=output_dir / 'correlation_matrix.html')

    figures['yearly_model_count'] = plot_yearly_model_count(
        df, save_path=output_dir / 'yearly_model_count.html')

    logger.info(f"All visualizations saved to {output_dir}")

    return figures


def run_full_analysis(df=None, save_outputs=True):
    """
    Run complete EDA pipeline

    Parameters:
    -----------
    df : pandas.DataFrame, optional
        Input dataframe (loads from file if not provided)
    save_outputs : bool
        Whether to save outputs to files

    Returns:
    --------
    dict : Analysis results including stats, insights, and figures
    """
    logger.info("=" * 50)
    logger.info("Starting Full EDA Analysis")
    logger.info("=" * 50)

    # Load data if not provided
    if df is None:
        df = load_cleaned_data()

    # Generate statistics
    stats = generate_summary_stats(df)

    # Generate insights
    insights = generate_insights_report(df)

    # Create visualizations
    figures = {}
    if save_outputs:
        figures = create_all_visualizations(df)

    results = {
        'statistics': stats,
        'insights': insights,
        'figures': figures,
        'data': df
    }

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("EDA Summary")
    logger.info("=" * 50)
    logger.info(f"Total models analyzed: {stats['total_models']}")
    logger.info(f"Unique tasks: {stats['unique_tasks']}")
    logger.info(f"Key insights: {len(insights['key_findings'])}")

    return results


if __name__ == "__main__":
    print("Starting Exploratory Data Analysis...")
    print("=" * 50)

    try:
        results = run_full_analysis(save_outputs=True)

        print("\nSummary Statistics:")
        for key, value in results['statistics'].items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")

        print("\nKey Insights:")
        for insight in results['insights']['key_findings']:
            print(f"  - {insight}")

        print("\nTrends:")
        for trend in results['insights']['trends']:
            print(f"  - {trend}")

    except FileNotFoundError:
        print("Cleaned data file not found. Please run the cleaner first.")
        print("Run: python src/cleaner.py")
