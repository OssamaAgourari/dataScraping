"""
Data Viewer Component for Streamlit Dashboard
Interactive data exploration and filtering
"""

import streamlit as st
import pandas as pd
import io


def render_data_viewer(df):
    """
    Render the data exploration page

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to explore
    """
    st.header("Data Explorer")

    if df is None:
        st.error("No data available. Please run the data pipeline first.")
        return

    # Filters section
    st.subheader("Filters")

    col1, col2, col3 = st.columns(3)

    # Create filter widgets
    filtered_df = df.copy()

    with col1:
        # Task filter
        if 'task' in df.columns:
            tasks = ['All'] + sorted(df['task'].unique().tolist())
            selected_task = st.selectbox("Select Task", tasks)
            if selected_task != 'All':
                filtered_df = filtered_df[filtered_df['task'] == selected_task]

    with col2:
        # Architecture filter
        arch_col = 'architecture_family' if 'architecture_family' in df.columns else 'architecture'
        if arch_col in df.columns:
            architectures = ['All'] + sorted(df[arch_col].dropna().unique().tolist())
            selected_arch = st.selectbox("Select Architecture", architectures)
            if selected_arch != 'All':
                filtered_df = filtered_df[filtered_df[arch_col] == selected_arch]

    with col3:
        # Year filter
        if 'year' in df.columns:
            years = df['year'].dropna().astype(int).unique()
            if len(years) > 0:
                year_range = st.slider(
                    "Year Range",
                    min_value=int(min(years)),
                    max_value=int(max(years)),
                    value=(int(min(years)), int(max(years)))
                )
                filtered_df = filtered_df[
                    (filtered_df['year'] >= year_range[0]) &
                    (filtered_df['year'] <= year_range[1])
                ]

    # Score filter
    if 'score' in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            min_score = st.number_input(
                "Minimum Score",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0
            )
        with col2:
            max_score = st.number_input(
                "Maximum Score",
                min_value=0.0,
                max_value=100.0,
                value=100.0,
                step=1.0
            )
        filtered_df = filtered_df[
            (filtered_df['score'] >= min_score) &
            (filtered_df['score'] <= max_score)
        ]

    # Search functionality
    search_term = st.text_input("Search Model Name", "")
    if search_term and 'model_name' in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df['model_name'].str.contains(search_term, case=False, na=False)
        ]

    st.divider()

    # Display filtered data info
    st.subheader(f"Results ({len(filtered_df):,} records)")

    # Column selection
    with st.expander("Select Columns to Display"):
        all_columns = filtered_df.columns.tolist()
        default_columns = ['model_name', 'task', 'score', 'year', 'architecture_family']
        default_columns = [col for col in default_columns if col in all_columns]

        selected_columns = st.multiselect(
            "Columns",
            options=all_columns,
            default=default_columns if default_columns else all_columns[:5]
        )

    # Sorting options
    col1, col2 = st.columns(2)
    with col1:
        sort_column = st.selectbox(
            "Sort by",
            options=['score', 'year', 'model_name'] if all(c in filtered_df.columns for c in ['score', 'year', 'model_name']) else filtered_df.columns.tolist()
        )
    with col2:
        sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)

    # Apply sorting
    ascending = sort_order == "Ascending"
    if sort_column in filtered_df.columns:
        filtered_df = filtered_df.sort_values(sort_column, ascending=ascending)

    # Display data table
    if selected_columns:
        display_df = filtered_df[selected_columns].reset_index(drop=True)
    else:
        display_df = filtered_df.reset_index(drop=True)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )

    st.divider()

    # Summary statistics
    st.subheader("Summary Statistics")

    col1, col2 = st.columns(2)

    with col1:
        if 'score' in filtered_df.columns:
            st.markdown("**Score Statistics:**")
            score_stats = filtered_df['score'].describe()
            stats_df = pd.DataFrame({
                'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                'Value': [
                    f"{score_stats['count']:.0f}",
                    f"{score_stats['mean']:.2f}",
                    f"{score_stats['std']:.2f}",
                    f"{score_stats['min']:.2f}",
                    f"{score_stats['25%']:.2f}",
                    f"{score_stats['50%']:.2f}",
                    f"{score_stats['75%']:.2f}",
                    f"{score_stats['max']:.2f}"
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**Category Counts:**")
        if 'task' in filtered_df.columns:
            st.write(f"- Tasks: {filtered_df['task'].nunique()}")
        if 'architecture_family' in filtered_df.columns:
            st.write(f"- Architectures: {filtered_df['architecture_family'].nunique()}")
        if 'dataset' in filtered_df.columns:
            st.write(f"- Datasets: {filtered_df['dataset'].nunique()}")
        if 'year' in filtered_df.columns:
            st.write(f"- Years: {filtered_df['year'].nunique()}")

    st.divider()

    # Download section
    st.subheader("Download Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        # CSV download
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name="ai_models_data.csv",
            mime="text/csv"
        )

    with col2:
        # JSON download
        json_data = filtered_df.to_json(orient='records', indent=2)

        st.download_button(
            label="Download as JSON",
            data=json_data,
            file_name="ai_models_data.json",
            mime="application/json"
        )

    with col3:
        # Excel download (requires openpyxl)
        try:
            excel_buffer = io.BytesIO()
            filtered_df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_data = excel_buffer.getvalue()

            st.download_button(
                label="Download as Excel",
                data=excel_data,
                file_name="ai_models_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception:
            st.info("Excel export requires openpyxl package")
