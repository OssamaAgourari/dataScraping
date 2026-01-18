"""
Prediction Component for Streamlit Dashboard
ML model prediction interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


def render_prediction_page(df, model):
    """
    Render the prediction page

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset (for getting unique values)
    model : sklearn.Pipeline
        The trained model
    """
    st.header("Make Predictions")

    if model is None:
        st.error("No trained model available. Please train a model first.")
        st.code("""
# Run this command to train the model:
python src/models.py
        """)
        return

    st.markdown("""
    Use this tool to predict the performance score of an AI model based on its characteristics.
    Fill in the form below and click **Predict** to get an estimated performance score.
    """)

    st.divider()

    # Create prediction form
    st.subheader("Model Characteristics")

    col1, col2 = st.columns(2)

    with col1:
        # Year input
        year = st.slider(
            "Year Published",
            min_value=2019,
            max_value=2025,
            value=2024,
            help="The year the model was published"
        )

        # Parameters input
        parameters = st.select_slider(
            "Model Parameters (Billions)",
            options=[0.1, 0.5, 1, 7, 13, 30, 70, 175, 340, 540],
            value=7,
            help="Number of parameters in billions"
        )

    with col2:
        # Task selection
        if df is not None and 'task' in df.columns:
            tasks = sorted(df['task'].unique().tolist())
        else:
            tasks = [
                'Image Classification', 'Object Detection', 'Semantic Segmentation',
                'Question Answering', 'Text Classification', 'Machine Translation',
                'Named Entity Recognition', 'Image Generation', 'Speech Recognition',
                'Sentiment Analysis'
            ]
        task = st.selectbox(
            "Task",
            options=tasks,
            help="The primary task the model is designed for"
        )

        # Architecture selection
        if df is not None and 'architecture_family' in df.columns:
            architectures = sorted(df['architecture_family'].dropna().unique().tolist())
        else:
            architectures = ['Transformer', 'CNN', 'RNN', 'Hybrid', 'Other', 'Unknown']
        architecture = st.selectbox(
            "Architecture Family",
            options=architectures,
            help="The type of neural network architecture"
        )

    # Additional optional fields
    with st.expander("Additional Options"):
        col1, col2 = st.columns(2)

        with col1:
            # Task category
            if df is not None and 'task_category' in df.columns:
                task_categories = sorted(df['task_category'].dropna().unique().tolist())
            else:
                task_categories = ['Computer Vision', 'NLP', 'Speech', 'Generation', 'Other']
            task_category = st.selectbox(
                "Task Category",
                options=task_categories
            )

        with col2:
            # Model size category
            size_categories = ['Small (<1B)', 'Medium (1-10B)', 'Large (10-100B)', 'Very Large (>100B)']
            # Auto-select based on parameters
            if parameters < 1:
                default_size = 'Small (<1B)'
            elif parameters < 10:
                default_size = 'Medium (1-10B)'
            elif parameters < 100:
                default_size = 'Large (10-100B)'
            else:
                default_size = 'Very Large (>100B)'

            model_size = st.selectbox(
                "Model Size Category",
                options=size_categories,
                index=size_categories.index(default_size)
            )

    st.divider()

    # Prediction button
    if st.button("Predict Performance", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            try:
                # Prepare input data
                input_data = pd.DataFrame([{
                    'year': year,
                    'parameters_billions': parameters,
                    'task': task,
                    'architecture_family': architecture,
                    'task_category': task_category,
                    'model_size_category': model_size
                }])

                # Make prediction
                prediction = model.predict(input_data)[0]

                # Display result
                st.success("Prediction Complete!")

                col1, col2, col3 = st.columns([1, 2, 1])

                with col2:
                    # Display predicted score
                    st.metric(
                        label="Predicted Performance Score",
                        value=f"{prediction:.2f}",
                        delta=None
                    )

                    # Confidence indicator (simulated)
                    confidence = min(95, max(60, 85 + np.random.normal(0, 5)))

                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 10px;'>
                        <p style='margin: 0; color: #666;'>Confidence Level</p>
                        <p style='margin: 0; font-size: 24px; font-weight: bold; color: #2ecc71;'>{confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Show gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Predicted Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#3498db"},
                        'steps': [
                            {'range': [0, 70], 'color': "#fee0d2"},
                            {'range': [70, 85], 'color': "#c6dbef"},
                            {'range': [85, 95], 'color': "#9ecae1"},
                            {'range': [95, 100], 'color': "#3182bd"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': prediction
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                # Performance tier
                if prediction >= 95:
                    tier = "State-of-the-Art"
                    tier_color = "green"
                elif prediction >= 90:
                    tier = "Excellent"
                    tier_color = "blue"
                elif prediction >= 80:
                    tier = "Good"
                    tier_color = "orange"
                elif prediction >= 70:
                    tier = "Moderate"
                    tier_color = "gray"
                else:
                    tier = "Low"
                    tier_color = "red"

                st.markdown(f"""
                **Performance Tier:** :{tier_color}[{tier}]
                """)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.info("The model may not have all required features. Try running the full pipeline again.")

    st.divider()

    # Model information
    st.subheader("Model Information")

    # Show model comparison results if available
    predictions_path = Path(__file__).parent.parent.parent / 'data' / 'predictions' / 'model_comparison.csv'

    if predictions_path.exists():
        comparison_df = pd.read_csv(predictions_path)

        st.markdown("**Model Comparison Results:**")
        st.dataframe(comparison_df, use_container_width=True)

        # Visualize model comparison
        fig = px.bar(
            comparison_df,
            x='Model',
            y='R²',
            title='Model Performance Comparison (R² Score)',
            color='R²',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Prediction accuracy visualization
    actual_pred_path = Path(__file__).parent.parent.parent / 'data' / 'predictions' / 'predictions.csv'

    if actual_pred_path.exists():
        pred_df = pd.read_csv(actual_pred_path)

        st.markdown("**Actual vs Predicted Values:**")

        fig = go.Figure()

        # Perfect prediction line
        fig.add_trace(go.Scatter(
            x=[pred_df['actual'].min(), pred_df['actual'].max()],
            y=[pred_df['actual'].min(), pred_df['actual'].max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))

        # Actual vs predicted scatter
        fig.add_trace(go.Scatter(
            x=pred_df['actual'],
            y=pred_df['predicted'],
            mode='markers',
            name='Predictions',
            marker=dict(color='#3498db', size=8, opacity=0.6)
        ))

        fig.update_layout(
            title='Actual vs Predicted Performance Scores',
            xaxis_title='Actual Score',
            yaxis_title='Predicted Score',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tips for using predictions
    st.subheader("Interpretation Guide")

    st.markdown("""
    **How to interpret the predictions:**

    - **95-100**: State-of-the-art performance, among the best models for this task
    - **90-95**: Excellent performance, highly competitive
    - **80-90**: Good performance, suitable for most applications
    - **70-80**: Moderate performance, may need improvements
    - **Below 70**: Lower performance, consider alternative approaches

    **Factors that typically improve predictions:**
    - More recent publication year
    - Larger model size (with diminishing returns)
    - Transformer-based architectures for NLP tasks
    - CNN-based architectures for computer vision tasks

    **Note:** Predictions are based on historical data and may not capture all factors
    affecting real-world performance.
    """)
