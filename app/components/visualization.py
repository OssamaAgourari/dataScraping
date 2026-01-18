"""
Visualization Component for Streamlit Dashboard
Interactive charts and graphs for data analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_visualizations(df):
    """
    Render the visualizations page

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to visualize
    """
    st.header("Data Visualizations")

    if df is None:
        st.error("No data available. Please run the data pipeline first.")
        return

    # Create tabs for different visualization categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "Performance Overview",
        "Trends & Time Series",
        "Comparisons",
        "Correlations"
    ])

    with tab1:
        render_performance_overview(df)

    with tab2:
        render_trends(df)

    with tab3:
        render_comparisons(df)

    with tab4:
        render_correlations(df)


def render_performance_overview(df):
    """Render performance overview visualizations"""

    st.subheader("Performance Overview")

    col1, col2 = st.columns(2)

    with col1:
        # Score distribution histogram
        if 'score' in df.columns:
            fig = px.histogram(
                df, x='score', nbins=40,
                title='Performance Score Distribution',
                color_discrete_sequence=['#3498db'],
                labels={'score': 'Performance Score', 'count': 'Number of Models'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Box plot of scores
        if 'score' in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=df['score'],
                name='Scores',
                marker_color='#2ecc71',
                boxpoints='outliers'
            ))
            fig.update_layout(
                title='Score Distribution (Box Plot)',
                yaxis_title='Performance Score',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    # Top performing models
    st.subheader("Top 15 Performing Models")

    if 'score' in df.columns and 'model_name' in df.columns:
        top_n = 15
        top_models = df.nlargest(top_n, 'score')

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_models['model_name'],
            x=top_models['score'],
            orientation='h',
            marker_color=px.colors.sequential.Blues_r[:len(top_models)],
            text=top_models['score'].round(2),
            textposition='outside'
        ))
        fig.update_layout(
            title=f'Top {top_n} Models by Performance Score',
            xaxis_title='Performance Score',
            yaxis_title='Model',
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)


def render_trends(df):
    """Render time series and trend visualizations"""

    st.subheader("Performance Trends Over Time")

    if 'year' not in df.columns or 'score' not in df.columns:
        st.warning("Year or score data not available for trend analysis")
        return

    # Yearly performance trends
    yearly_stats = df.groupby('year').agg({
        'score': ['mean', 'max', 'min', 'count', 'std']
    }).round(2)
    yearly_stats.columns = ['mean_score', 'max_score', 'min_score', 'count', 'std_score']
    yearly_stats = yearly_stats.reset_index()

    # Line chart with confidence band
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

    # Add confidence band
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
        title='Performance Trends Over Time',
        xaxis_title='Year',
        yaxis_title='Performance Score',
        height=450,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model count over years
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            yearly_stats,
            x='year', y='count',
            title='Number of Models Published Per Year',
            color='count',
            color_continuous_scale='Purples'
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Standard deviation over time (shows if field is converging)
        fig = px.line(
            yearly_stats,
            x='year', y='std_score',
            title='Score Variance Over Time',
            markers=True
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


def render_comparisons(df):
    """Render comparison visualizations"""

    st.subheader("Performance Comparisons")

    # Task comparison
    if 'task' in df.columns and 'score' in df.columns:
        st.markdown("**Performance by Task**")

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
            textposition='outside'
        ))
        fig.update_layout(
            title='Mean Performance Score by Task',
            xaxis_title='Mean Performance Score',
            yaxis_title='Task',
            height=500,
            margin=dict(l=200)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Architecture comparison
    arch_col = 'architecture_family' if 'architecture_family' in df.columns else 'architecture'
    if arch_col in df.columns and 'score' in df.columns:
        st.markdown("**Performance by Architecture**")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.box(
                df,
                x=arch_col,
                y='score',
                color=arch_col,
                title='Score Distribution by Architecture'
            )
            fig.update_layout(height=400, showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            arch_counts = df[arch_col].value_counts()
            fig = px.pie(
                values=arch_counts.values,
                names=arch_counts.index,
                title='Model Distribution by Architecture'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Performance tier breakdown
    if 'performance_tier' in df.columns:
        st.markdown("**Performance Tier Distribution**")

        tier_counts = df['performance_tier'].value_counts()
        fig = px.bar(
            x=tier_counts.index,
            y=tier_counts.values,
            title='Number of Models by Performance Tier',
            color=tier_counts.index,
            color_discrete_sequence=px.colors.sequential.RdYlGn
        )
        fig.update_layout(
            xaxis_title='Performance Tier',
            yaxis_title='Number of Models',
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)


def render_correlations(df):
    """Render correlation analysis visualizations"""

    st.subheader("Correlation Analysis")

    # Model size vs performance
    if 'parameters_billions' in df.columns and 'score' in df.columns:
        st.markdown("**Model Size vs Performance**")

        plot_df = df.dropna(subset=['parameters_billions', 'score'])

        if len(plot_df) > 0:
            fig = px.scatter(
                plot_df,
                x='parameters_billions',
                y='score',
                color='task' if 'task' in plot_df.columns else None,
                size='score',
                hover_name='model_name' if 'model_name' in plot_df.columns else None,
                title='Model Parameters vs Performance Score',
                labels={
                    'parameters_billions': 'Parameters (Billions)',
                    'score': 'Performance Score'
                },
                log_x=True
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Calculate and display correlation
            corr = plot_df['parameters_billions'].corr(plot_df['score'])
            st.info(f"Correlation coefficient: {corr:.3f}")
        else:
            st.warning("Not enough data for size vs performance analysis")

    # Correlation matrix
    st.markdown("**Feature Correlation Matrix**")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title='Feature Correlation Heatmap',
            aspect='auto'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough numeric columns for correlation analysis")

    # Year vs Score scatter
    if 'year' in df.columns and 'score' in df.columns:
        st.markdown("**Score Distribution Over Time**")

        fig = px.scatter(
            df,
            x='year',
            y='score',
            color='task' if 'task' in df.columns else None,
            opacity=0.6,
            title='Individual Model Scores Over Time'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
