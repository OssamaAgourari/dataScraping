"""
Data Cleaning Module for AI Model Performance Analysis
Handles data preprocessing, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_raw_data(filepath=None):
    """
    Load raw scraped data from CSV file

    Parameters:
    -----------
    filepath : str or Path, optional
        Path to the raw data CSV file

    Returns:
    --------
    pandas.DataFrame : Raw data
    """
    if filepath is None:
        filepath = Path(__file__).parent.parent / 'data' / 'raw' / 'scraped_data.csv'

    logger.info(f"Loading raw data from {filepath}")

    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def handle_missing_values(df):
    """
    Handle missing values in the dataset

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe

    Returns:
    --------
    pandas.DataFrame : Dataframe with handled missing values
    """
    logger.info("Handling missing values...")

    # Report missing values
    missing_report = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)

    for col in df.columns:
        if missing_report[col] > 0:
            logger.info(f"  {col}: {missing_report[col]} missing ({missing_pct[col]}%)")

    # Strategy for each column type
    # Drop rows with missing model_name (essential field)
    if 'model_name' in df.columns:
        initial_count = len(df)
        df = df.dropna(subset=['model_name'])
        dropped = initial_count - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with missing model_name")

    # Fill missing scores with NaN (will be handled later or dropped)
    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')

    # Fill missing categorical values with 'Unknown'
    categorical_cols = ['task', 'dataset', 'architecture', 'metric']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # Fill missing dates with most common year or median
    if 'date' in df.columns:
        df['date'] = df['date'].fillna(df['date'].mode().iloc[0] if not df['date'].mode().empty else '2023')

    # Fill missing boolean values
    if 'extra_training_data' in df.columns:
        df['extra_training_data'] = df['extra_training_data'].fillna(False)

    return df


def clean_text_columns(df):
    """
    Standardize and clean text columns

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe

    Returns:
    --------
    pandas.DataFrame : Dataframe with cleaned text columns
    """
    logger.info("Cleaning text columns...")

    text_columns = ['model_name', 'paper_title', 'task', 'dataset', 'architecture']

    for col in text_columns:
        if col in df.columns:
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()

            # Remove excessive whitespace
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)

            # Replace common variations
            if col == 'model_name':
                # Standardize model name formats
                df[col] = df[col].str.replace(r'\s*-\s*', '-', regex=True)
                df[col] = df[col].str.replace(r'\s*/\s*', '/', regex=True)

            if col == 'task':
                # Standardize task names
                df[col] = df[col].str.lower().str.replace('_', ' ').str.title()

    return df


def extract_numeric_features(df):
    """
    Extract and convert numeric features from text

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe

    Returns:
    --------
    pandas.DataFrame : Dataframe with extracted numeric features
    """
    logger.info("Extracting numeric features...")

    # Convert score to numeric
    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')

        # Handle scores that might be in decimal form (0-1) vs percentage (0-100)
        mask = (df['score'] <= 1) & (df['score'] > 0)
        df.loc[mask, 'score'] = df.loc[mask, 'score'] * 100

    # Extract parameter count from text (e.g., "7B" -> 7000000000)
    if 'parameters' in df.columns:
        df['parameters_raw'] = df['parameters'].copy()
        df['parameters_billions'] = df['parameters'].apply(parse_parameter_count)

    # Extract year from date
    if 'date' in df.columns:
        df['year'] = df['date'].apply(extract_year)

    return df


def parse_parameter_count(param_str):
    """
    Parse parameter count from string format (e.g., "7B", "350M", "1.5B")

    Parameters:
    -----------
    param_str : str
        Parameter count string

    Returns:
    --------
    float : Parameter count in billions, or NaN if unparseable
    """
    if pd.isna(param_str) or param_str == 'None' or param_str == 'nan':
        return np.nan

    param_str = str(param_str).upper().strip()

    # Match patterns like "7B", "350M", "1.5B", "7 billion"
    patterns = [
        (r'([\d.]+)\s*B', 1),           # Billions
        (r'([\d.]+)\s*M', 0.001),       # Millions -> Billions
        (r'([\d.]+)\s*K', 0.000001),    # Thousands -> Billions
        (r'([\d.]+)\s*BILLION', 1),
        (r'([\d.]+)\s*MILLION', 0.001),
    ]

    for pattern, multiplier in patterns:
        match = re.search(pattern, param_str)
        if match:
            try:
                return float(match.group(1)) * multiplier
            except ValueError:
                continue

    # Try to extract just a number
    match = re.search(r'([\d.]+)', param_str)
    if match:
        try:
            val = float(match.group(1))
            # Assume it's in billions if > 1000, millions otherwise
            if val > 1000:
                return val / 1000
            return val
        except ValueError:
            pass

    return np.nan


def extract_year(date_str):
    """
    Extract year from various date formats

    Parameters:
    -----------
    date_str : str
        Date string

    Returns:
    --------
    int : Year, or NaN if unparseable
    """
    if pd.isna(date_str) or date_str == 'None' or date_str == 'nan':
        return np.nan

    date_str = str(date_str)

    # Try to find a 4-digit year
    match = re.search(r'(20\d{2}|19\d{2})', date_str)
    if match:
        return int(match.group(1))

    return np.nan


def remove_duplicates(df):
    """
    Remove duplicate entries

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe

    Returns:
    --------
    pandas.DataFrame : Dataframe with duplicates removed
    """
    logger.info("Removing duplicates...")

    initial_count = len(df)

    # Define subset of columns to check for duplicates
    subset_cols = ['model_name', 'task', 'dataset']
    subset_cols = [col for col in subset_cols if col in df.columns]

    if subset_cols:
        # Keep the entry with the highest score (most recent/best result)
        if 'score' in df.columns:
            df = df.sort_values('score', ascending=False)

        df = df.drop_duplicates(subset=subset_cols, keep='first')

    removed = initial_count - len(df)
    logger.info(f"Removed {removed} duplicate entries")

    return df


def engineer_features(df):
    """
    Create new features from existing data

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe

    Returns:
    --------
    pandas.DataFrame : Dataframe with engineered features
    """
    logger.info("Engineering features...")

    # Create parameter size category
    if 'parameters_billions' in df.columns:
        df['model_size_category'] = pd.cut(
            df['parameters_billions'],
            bins=[0, 1, 10, 100, float('inf')],
            labels=['Small (<1B)', 'Medium (1-10B)', 'Large (10-100B)', 'Very Large (>100B)']
        )

    # Create year bins
    if 'year' in df.columns:
        current_year = datetime.now().year
        df['era'] = pd.cut(
            df['year'],
            bins=[2015, 2019, 2021, 2023, current_year + 1],
            labels=['Pre-Transformer (2016-2019)', 'Early Transformer (2020-2021)',
                    'LLM Era (2022-2023)', 'Current (2024+)']
        )

    # Create score bins
    if 'score' in df.columns:
        df['performance_tier'] = pd.cut(
            df['score'],
            bins=[0, 70, 80, 90, 95, 100],
            labels=['Low', 'Moderate', 'Good', 'Excellent', 'State-of-the-Art']
        )

    # Architecture family grouping
    if 'architecture' in df.columns:
        arch_mapping = {
            'Transformer': ['Transformer', 'BERT', 'GPT', 'T5', 'LLaMA', 'ViT', 'CLIP'],
            'CNN': ['CNN', 'ResNet', 'EfficientNet', 'ConvNeXt', 'DenseNet', 'VGG', 'Inception'],
            'RNN': ['LSTM', 'GRU', 'RNN', 'Seq2Seq'],
            'Hybrid': ['Hybrid', 'Mixed', 'Ensemble'],
            'Other': []
        }

        def map_architecture_family(arch):
            if pd.isna(arch) or arch == 'Unknown':
                return 'Unknown'
            arch_upper = str(arch).upper()
            for family, keywords in arch_mapping.items():
                for keyword in keywords:
                    if keyword.upper() in arch_upper:
                        return family
            return 'Other'

        df['architecture_family'] = df['architecture'].apply(map_architecture_family)

    # Task category grouping
    if 'task' in df.columns:
        task_mapping = {
            'Computer Vision': ['image', 'object', 'segmentation', 'detection', 'visual'],
            'NLP': ['text', 'question', 'translation', 'entity', 'sentiment', 'language'],
            'Speech': ['speech', 'audio', 'recognition'],
            'Generation': ['generation', 'synthesis', 'generative'],
        }

        def map_task_category(task):
            if pd.isna(task):
                return 'Other'
            task_lower = str(task).lower()
            for category, keywords in task_mapping.items():
                for keyword in keywords:
                    if keyword in task_lower:
                        return category
            return 'Other'

        df['task_category'] = df['task'].apply(map_task_category)

    return df


def validate_data(df):
    """
    Validate cleaned data for quality and consistency

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe

    Returns:
    --------
    tuple : (bool, list) - Validation status and list of issues
    """
    logger.info("Validating data...")

    issues = []

    # Check for required columns
    required_cols = ['model_name', 'task', 'score']
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")

    # Check score range
    if 'score' in df.columns:
        invalid_scores = df[(df['score'] < 0) | (df['score'] > 100)]['score']
        if len(invalid_scores) > 0:
            issues.append(f"Found {len(invalid_scores)} scores outside valid range (0-100)")

    # Check year range
    if 'year' in df.columns:
        current_year = datetime.now().year
        invalid_years = df[(df['year'] < 2010) | (df['year'] > current_year + 1)]['year']
        if len(invalid_years) > 0:
            issues.append(f"Found {len(invalid_years)} years outside valid range")

    # Check for empty dataframe
    if len(df) == 0:
        issues.append("Dataframe is empty")

    # Report validation results
    if issues:
        for issue in issues:
            logger.warning(f"Validation issue: {issue}")
        return False, issues
    else:
        logger.info("Data validation passed")
        return True, []


def save_cleaned_data(df, filepath=None):
    """
    Save cleaned data to CSV file

    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned dataframe
    filepath : str or Path, optional
        Output filepath
    """
    if filepath is None:
        filepath = Path(__file__).parent.parent / 'data' / 'processed' / 'cleaned_data.csv'

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(filepath, index=False)
    logger.info(f"Cleaned data saved to {filepath}")

    return filepath


def clean_data(input_path=None, output_path=None):
    """
    Main cleaning pipeline - runs all cleaning steps

    Parameters:
    -----------
    input_path : str or Path, optional
        Path to raw data
    output_path : str or Path, optional
        Path to save cleaned data

    Returns:
    --------
    pandas.DataFrame : Cleaned dataframe
    """
    logger.info("=" * 50)
    logger.info("Starting data cleaning pipeline")
    logger.info("=" * 50)

    # Load data
    df = load_raw_data(input_path)

    # Store original shape for reporting
    original_shape = df.shape
    logger.info(f"Original data shape: {original_shape}")

    # Run cleaning steps
    df = handle_missing_values(df)
    df = clean_text_columns(df)
    df = extract_numeric_features(df)
    df = remove_duplicates(df)
    df = engineer_features(df)

    # Validate
    is_valid, issues = validate_data(df)

    # Save cleaned data
    output_file = save_cleaned_data(df, output_path)

    # Report summary
    logger.info("=" * 50)
    logger.info("Cleaning Summary:")
    logger.info(f"  Original records: {original_shape[0]}")
    logger.info(f"  Cleaned records: {len(df)}")
    logger.info(f"  Records removed: {original_shape[0] - len(df)}")
    logger.info(f"  Final columns: {list(df.columns)}")
    logger.info(f"  Output file: {output_file}")
    logger.info("=" * 50)

    return df


def get_cleaning_report(df_raw, df_cleaned):
    """
    Generate a detailed cleaning report

    Parameters:
    -----------
    df_raw : pandas.DataFrame
        Raw dataframe
    df_cleaned : pandas.DataFrame
        Cleaned dataframe

    Returns:
    --------
    dict : Cleaning report
    """
    report = {
        'original_rows': len(df_raw),
        'cleaned_rows': len(df_cleaned),
        'rows_removed': len(df_raw) - len(df_cleaned),
        'removal_percentage': round((len(df_raw) - len(df_cleaned)) / len(df_raw) * 100, 2),
        'original_columns': list(df_raw.columns),
        'new_columns': [col for col in df_cleaned.columns if col not in df_raw.columns],
        'missing_values_before': df_raw.isnull().sum().to_dict(),
        'missing_values_after': df_cleaned.isnull().sum().to_dict(),
        'dtypes': df_cleaned.dtypes.astype(str).to_dict(),
    }

    return report


if __name__ == "__main__":
    # Run cleaning pipeline
    print("Starting Data Cleaning Pipeline...")
    print("=" * 50)

    try:
        df_cleaned = clean_data()

        print("\nCleaned Data Preview:")
        print(df_cleaned.head(10).to_string())

        print("\nData Types:")
        print(df_cleaned.dtypes)

        print("\nColumn Statistics:")
        print(df_cleaned.describe())

    except FileNotFoundError:
        print("Raw data file not found. Please run the scraper first.")
        print("Run: python src/scraper.py")
