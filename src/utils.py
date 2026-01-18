"""
Utility Functions for AI Model Performance Analysis
Common helper functions used across modules
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from pathlib import Path
from datetime import datetime
import hashlib

# Setup logging
def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Setup logging configuration

    Parameters:
    -----------
    log_level : int
        Logging level (e.g., logging.INFO, logging.DEBUG)
    log_file : str or Path, optional
        Path to log file

    Returns:
    --------
    logging.Logger : Configured logger
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers = []

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_project_root():
    """
    Get the project root directory

    Returns:
    --------
    Path : Project root path
    """
    return Path(__file__).parent.parent


def get_data_path(data_type='processed'):
    """
    Get path to data directory

    Parameters:
    -----------
    data_type : str
        Type of data ('raw', 'processed', 'predictions')

    Returns:
    --------
    Path : Data directory path
    """
    root = get_project_root()
    return root / 'data' / data_type


def validate_data(df, required_columns=None, check_types=True):
    """
    Validate dataframe structure and content

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe to validate
    required_columns : list, optional
        List of required column names
    check_types : bool
        Whether to check data types

    Returns:
    --------
    tuple : (is_valid, issues_list)
    """
    issues = []

    # Check if dataframe is empty
    if df is None or len(df) == 0:
        issues.append("Dataframe is empty or None")
        return False, issues

    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")

    # Check for all-null columns
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        issues.append(f"Columns with all null values: {null_columns}")

    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate rows")

    # Return validation result
    is_valid = len(issues) == 0
    return is_valid, issues


def calculate_data_quality_score(df):
    """
    Calculate a data quality score for the dataframe

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe

    Returns:
    --------
    dict : Quality metrics
    """
    quality = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'completeness': {},
        'overall_completeness': 0,
        'duplicate_rate': 0,
        'quality_score': 0
    }

    # Calculate completeness for each column
    for col in df.columns:
        non_null_rate = 1 - (df[col].isnull().sum() / len(df))
        quality['completeness'][col] = round(non_null_rate * 100, 2)

    # Overall completeness
    quality['overall_completeness'] = round(
        (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2
    )

    # Duplicate rate
    quality['duplicate_rate'] = round(
        df.duplicated().sum() / len(df) * 100, 2
    )

    # Calculate overall quality score (weighted average)
    quality['quality_score'] = round(
        quality['overall_completeness'] * 0.6 +
        (100 - quality['duplicate_rate']) * 0.4, 2
    )

    return quality


def export_data(df, filepath, format='csv'):
    """
    Export dataframe to various formats

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe to export
    filepath : str or Path
        Output file path
    format : str
        Export format ('csv', 'excel', 'json', 'parquet')

    Returns:
    --------
    Path : Path to exported file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'excel':
        df.to_excel(filepath, index=False, engine='openpyxl')
    elif format == 'json':
        df.to_json(filepath, orient='records', indent=2)
    elif format == 'parquet':
        df.to_parquet(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logging.info(f"Data exported to {filepath}")
    return filepath


def load_config(config_path=None):
    """
    Load configuration from JSON file

    Parameters:
    -----------
    config_path : str or Path, optional
        Path to config file

    Returns:
    --------
    dict : Configuration dictionary
    """
    if config_path is None:
        config_path = get_project_root() / 'config.json'

    config_path = Path(config_path)

    if not config_path.exists():
        # Return default config
        return {
            'scraping': {
                'delay': 2,
                'max_retries': 3,
                'timeout': 30
            },
            'modeling': {
                'test_size': 0.2,
                'random_state': 42,
                'cv_folds': 5
            },
            'visualization': {
                'theme': 'plotly_white',
                'color_palette': 'Set2'
            }
        }

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def save_config(config, config_path=None):
    """
    Save configuration to JSON file

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    config_path : str or Path, optional
        Path to save config file
    """
    if config_path is None:
        config_path = get_project_root() / 'config.json'

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logging.info(f"Config saved to {config_path}")


def generate_report_id():
    """
    Generate a unique report ID based on timestamp

    Returns:
    --------
    str : Unique report ID
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    hash_suffix = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:6]
    return f"report_{timestamp}_{hash_suffix}"


def format_number(num, precision=2):
    """
    Format number for display

    Parameters:
    -----------
    num : float or int
        Number to format
    precision : int
        Decimal precision

    Returns:
    --------
    str : Formatted number string
    """
    if pd.isna(num):
        return "N/A"

    if abs(num) >= 1e9:
        return f"{num/1e9:.{precision}f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def format_percentage(num, precision=1):
    """
    Format number as percentage

    Parameters:
    -----------
    num : float
        Number to format (0-1 or 0-100)
    precision : int
        Decimal precision

    Returns:
    --------
    str : Formatted percentage string
    """
    if pd.isna(num):
        return "N/A"

    # Assume 0-1 range if value is <= 1, otherwise 0-100
    if num <= 1:
        num = num * 100

    return f"{num:.{precision}f}%"


def truncate_text(text, max_length=50, suffix='...'):
    """
    Truncate text to maximum length

    Parameters:
    -----------
    text : str
        Text to truncate
    max_length : int
        Maximum length
    suffix : str
        Suffix to add if truncated

    Returns:
    --------
    str : Truncated text
    """
    if pd.isna(text):
        return ""

    text = str(text)
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def calculate_metrics_summary(df, score_col='score', group_col=None):
    """
    Calculate summary metrics for scores

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    score_col : str
        Name of score column
    group_col : str, optional
        Column to group by

    Returns:
    --------
    pandas.DataFrame : Summary metrics
    """
    if group_col and group_col in df.columns:
        summary = df.groupby(group_col)[score_col].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(2)
    else:
        summary = pd.DataFrame({
            'count': [len(df)],
            'mean': [df[score_col].mean()],
            'std': [df[score_col].std()],
            'min': [df[score_col].min()],
            'max': [df[score_col].max()],
            'median': [df[score_col].median()]
        }).round(2)

    return summary


def create_sample_input(feature_names, feature_ranges=None):
    """
    Create a sample input dataframe for prediction

    Parameters:
    -----------
    feature_names : list
        List of feature names
    feature_ranges : dict, optional
        Dictionary of feature ranges/options

    Returns:
    --------
    pandas.DataFrame : Sample input dataframe
    """
    if feature_ranges is None:
        feature_ranges = {
            'year': [2020, 2021, 2022, 2023, 2024],
            'parameters_billions': [0.1, 1, 10, 100],
            'task': ['Image Classification', 'Object Detection', 'Text Classification', 'Question Answering'],
            'architecture_family': ['Transformer', 'CNN', 'RNN', 'Hybrid'],
            'task_category': ['Computer Vision', 'NLP', 'Speech', 'Generation'],
            'model_size_category': ['Small (<1B)', 'Medium (1-10B)', 'Large (10-100B)', 'Very Large (>100B)']
        }

    sample = {}
    for feature in feature_names:
        if feature in feature_ranges:
            sample[feature] = feature_ranges[feature][0]
        else:
            sample[feature] = None

    return pd.DataFrame([sample])


def get_recent_files(directory, pattern='*', n=5):
    """
    Get most recently modified files in a directory

    Parameters:
    -----------
    directory : str or Path
        Directory to search
    pattern : str
        Glob pattern for files
    n : int
        Number of files to return

    Returns:
    --------
    list : List of (filepath, modified_time) tuples
    """
    directory = Path(directory)

    if not directory.exists():
        return []

    files = list(directory.glob(pattern))
    files_with_time = [(f, f.stat().st_mtime) for f in files if f.is_file()]
    files_with_time.sort(key=lambda x: x[1], reverse=True)

    return files_with_time[:n]


def ensure_directories():
    """
    Ensure all required project directories exist
    """
    root = get_project_root()

    directories = [
        root / 'data' / 'raw',
        root / 'data' / 'processed',
        root / 'data' / 'predictions',
        root / 'data' / 'models',
        root / 'data' / 'visualizations',
        root / 'notebooks',
        root / 'src',
        root / 'app' / 'components',
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    logging.info("All directories created/verified")


if __name__ == "__main__":
    # Setup logging
    logger = setup_logging(log_level=logging.INFO)

    # Ensure directories exist
    ensure_directories()

    print("Utility functions loaded successfully!")
    print(f"Project root: {get_project_root()}")
