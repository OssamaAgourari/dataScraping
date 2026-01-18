"""
Machine Learning Models for AI Model Performance Prediction
Trains and evaluates models to predict AI model performance based on features
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")


def load_data(filepath=None):
    """
    Load cleaned data for modeling

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

    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} records")

    return df


def prepare_features(df, target_col='score'):
    """
    Prepare features for modeling

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Name of target column

    Returns:
    --------
    tuple : (X, y, feature_names, preprocessor)
    """
    logger.info("Preparing features for modeling...")

    # Define feature columns
    numeric_features = []
    categorical_features = []

    # Check available columns and add to appropriate list
    if 'year' in df.columns:
        numeric_features.append('year')

    if 'parameters_billions' in df.columns:
        numeric_features.append('parameters_billions')

    if 'task' in df.columns:
        categorical_features.append('task')

    if 'architecture_family' in df.columns:
        categorical_features.append('architecture_family')
    elif 'architecture' in df.columns:
        categorical_features.append('architecture')

    if 'task_category' in df.columns:
        categorical_features.append('task_category')

    if 'model_size_category' in df.columns:
        categorical_features.append('model_size_category')

    logger.info(f"Numeric features: {numeric_features}")
    logger.info(f"Categorical features: {categorical_features}")

    # Select features and target
    all_features = numeric_features + categorical_features

    if len(all_features) == 0:
        raise ValueError("No suitable features found in the data")

    # Filter rows with valid target
    df_model = df.dropna(subset=[target_col]).copy()

    # Prepare feature matrix and target
    X = df_model[all_features].copy()
    y = df_model[target_col].copy()

    # Create preprocessor
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target variable shape: {y.shape}")

    return X, y, all_features, preprocessor


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets

    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target variable
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed

    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def build_model_pipeline(model, preprocessor):
    """
    Build a complete model pipeline with preprocessing

    Parameters:
    -----------
    model : sklearn estimator
        The model to use
    preprocessor : sklearn transformer
        The preprocessing pipeline

    Returns:
    --------
    sklearn.Pipeline : Complete model pipeline
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline


def train_linear_model(X_train, y_train, preprocessor):
    """
    Train a linear regression model

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    preprocessor : sklearn transformer
        Preprocessing pipeline

    Returns:
    --------
    sklearn.Pipeline : Trained model pipeline
    """
    logger.info("Training Linear Regression model...")

    model = LinearRegression()
    pipeline = build_model_pipeline(model, preprocessor)
    pipeline.fit(X_train, y_train)

    return pipeline


def train_ridge_model(X_train, y_train, preprocessor, alpha=1.0):
    """
    Train a Ridge regression model

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    preprocessor : sklearn transformer
        Preprocessing pipeline
    alpha : float
        Regularization strength

    Returns:
    --------
    sklearn.Pipeline : Trained model pipeline
    """
    logger.info("Training Ridge Regression model...")

    model = Ridge(alpha=alpha)
    pipeline = build_model_pipeline(model, preprocessor)
    pipeline.fit(X_train, y_train)

    return pipeline


def train_random_forest(X_train, y_train, preprocessor, n_estimators=100, max_depth=10):
    """
    Train a Random Forest model

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    preprocessor : sklearn transformer
        Preprocessing pipeline
    n_estimators : int
        Number of trees
    max_depth : int
        Maximum tree depth

    Returns:
    --------
    sklearn.Pipeline : Trained model pipeline
    """
    logger.info("Training Random Forest model...")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    pipeline = build_model_pipeline(model, preprocessor)
    pipeline.fit(X_train, y_train)

    return pipeline


def train_gradient_boosting(X_train, y_train, preprocessor, n_estimators=100, max_depth=5):
    """
    Train a Gradient Boosting model

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    preprocessor : sklearn transformer
        Preprocessing pipeline
    n_estimators : int
        Number of boosting stages
    max_depth : int
        Maximum tree depth

    Returns:
    --------
    sklearn.Pipeline : Trained model pipeline
    """
    logger.info("Training Gradient Boosting model...")

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    pipeline = build_model_pipeline(model, preprocessor)
    pipeline.fit(X_train, y_train)

    return pipeline


def train_xgboost(X_train, y_train, preprocessor, n_estimators=100, max_depth=5):
    """
    Train an XGBoost model

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    preprocessor : sklearn transformer
        Preprocessing pipeline
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum tree depth

    Returns:
    --------
    sklearn.Pipeline : Trained model pipeline
    """
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost not available, skipping...")
        return None

    logger.info("Training XGBoost model...")

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        verbosity=0
    )
    pipeline = build_model_pipeline(model, preprocessor)
    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model

    Parameters:
    -----------
    model : sklearn.Pipeline
        Trained model pipeline
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
    model_name : str
        Name of the model for reporting

    Returns:
    --------
    dict : Evaluation metrics
    """
    logger.info(f"Evaluating {model_name}...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'model_name': model_name,
        'mse': round(mse, 4),
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'r2': round(r2, 4),
        'predictions': y_pred
    }

    logger.info(f"  RMSE: {metrics['rmse']}")
    logger.info(f"  MAE: {metrics['mae']}")
    logger.info(f"  R²: {metrics['r2']}")

    return metrics


def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross-validation on a model

    Parameters:
    -----------
    model : sklearn.Pipeline
        Model pipeline (untrained)
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Target
    cv : int
        Number of folds

    Returns:
    --------
    dict : Cross-validation results
    """
    logger.info(f"Performing {cv}-fold cross-validation...")

    # Calculate scores
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)

    results = {
        'cv_rmse_mean': round(rmse_scores.mean(), 4),
        'cv_rmse_std': round(rmse_scores.std(), 4),
        'cv_scores': rmse_scores.tolist()
    }

    logger.info(f"  CV RMSE: {results['cv_rmse_mean']} (+/- {results['cv_rmse_std']})")

    return results


def compare_models(models_dict, X_test, y_test):
    """
    Compare multiple models and return ranking

    Parameters:
    -----------
    models_dict : dict
        Dictionary of model names and trained models
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target

    Returns:
    --------
    pandas.DataFrame : Model comparison results
    """
    logger.info("Comparing models...")

    results = []

    for name, model in models_dict.items():
        if model is not None:
            metrics = evaluate_model(model, X_test, y_test, name)
            results.append({
                'Model': name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2']
            })

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('R²', ascending=False)

    logger.info("\nModel Comparison:")
    logger.info(comparison_df.to_string(index=False))

    return comparison_df


def get_feature_importance(model, feature_names, preprocessor):
    """
    Get feature importance from a trained model

    Parameters:
    -----------
    model : sklearn.Pipeline
        Trained model pipeline
    feature_names : list
        Original feature names
    preprocessor : sklearn transformer
        Preprocessor used in pipeline

    Returns:
    --------
    pandas.DataFrame : Feature importance scores
    """
    try:
        # Get the actual model from pipeline
        actual_model = model.named_steps['model']

        # Get feature names after transformation
        if hasattr(preprocessor, 'get_feature_names_out'):
            transformed_names = preprocessor.get_feature_names_out()
        else:
            transformed_names = feature_names

        # Get feature importance
        if hasattr(actual_model, 'feature_importances_'):
            importances = actual_model.feature_importances_
        elif hasattr(actual_model, 'coef_'):
            importances = np.abs(actual_model.coef_)
        else:
            logger.warning("Model doesn't have feature importance attribute")
            return None

        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': transformed_names[:len(importances)],
            'importance': importances
        })
        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df

    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
        return None


def save_model(model, filepath=None, model_name="best_model"):
    """
    Save a trained model to disk

    Parameters:
    -----------
    model : sklearn.Pipeline
        Trained model pipeline
    filepath : str or Path, optional
        Path to save the model
    model_name : str
        Name for the model file
    """
    if filepath is None:
        models_dir = Path(__file__).parent.parent / 'data' / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        filepath = models_dir / f"{model_name}.joblib"

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")

    return filepath


def load_model(filepath):
    """
    Load a trained model from disk

    Parameters:
    -----------
    filepath : str or Path
        Path to the saved model

    Returns:
    --------
    sklearn.Pipeline : Loaded model
    """
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")

    return model


def predict_performance(model, new_data):
    """
    Make predictions using a trained model

    Parameters:
    -----------
    model : sklearn.Pipeline
        Trained model pipeline
    new_data : pandas.DataFrame
        New data to predict

    Returns:
    --------
    numpy.ndarray : Predictions
    """
    predictions = model.predict(new_data)
    return predictions


def train_models(df=None, save_best=True):
    """
    Main function to train all models

    Parameters:
    -----------
    df : pandas.DataFrame, optional
        Input data (loads from file if not provided)
    save_best : bool
        Whether to save the best model

    Returns:
    --------
    dict : Training results including models and metrics
    """
    logger.info("=" * 50)
    logger.info("Starting Model Training Pipeline")
    logger.info("=" * 50)

    # Load data if not provided
    if df is None:
        df = load_data()

    # Prepare features
    X, y, feature_names, preprocessor = prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Train models
    models = {}

    models['Linear Regression'] = train_linear_model(X_train, y_train, preprocessor)
    models['Ridge Regression'] = train_ridge_model(X_train, y_train, preprocessor)
    models['Random Forest'] = train_random_forest(X_train, y_train, preprocessor)
    models['Gradient Boosting'] = train_gradient_boosting(X_train, y_train, preprocessor)

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = train_xgboost(X_train, y_train, preprocessor)

    # Compare models
    comparison_df = compare_models(models, X_test, y_test)

    # Get best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = models[best_model_name]

    logger.info(f"\nBest Model: {best_model_name}")

    # Get feature importance for best model
    feature_importance = get_feature_importance(best_model, feature_names, preprocessor)

    # Save best model
    if save_best:
        save_model(best_model, model_name=f"best_model_{best_model_name.lower().replace(' ', '_')}")

    # Save predictions
    predictions_dir = Path(__file__).parent.parent / 'data' / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': best_model.predict(X_test)
    })
    predictions_df.to_csv(predictions_dir / 'predictions.csv', index=False)

    # Save comparison results
    comparison_df.to_csv(predictions_dir / 'model_comparison.csv', index=False)

    results = {
        'models': models,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'comparison': comparison_df,
        'feature_importance': feature_importance,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'preprocessor': preprocessor
    }

    logger.info("=" * 50)
    logger.info("Model Training Complete!")
    logger.info("=" * 50)

    return results


if __name__ == "__main__":
    print("Starting Machine Learning Pipeline...")
    print("=" * 50)

    try:
        results = train_models(save_best=True)

        print("\n" + "=" * 50)
        print("Model Comparison Results:")
        print("=" * 50)
        print(results['comparison'].to_string(index=False))

        print("\n" + "=" * 50)
        print(f"Best Model: {results['best_model_name']}")
        print("=" * 50)

        if results['feature_importance'] is not None:
            print("\nTop 10 Feature Importances:")
            print(results['feature_importance'].head(10).to_string(index=False))

    except FileNotFoundError:
        print("Cleaned data file not found. Please run the cleaner first.")
        print("Run: python src/cleaner.py")
    except Exception as e:
        print(f"Error during model training: {e}")
        raise
