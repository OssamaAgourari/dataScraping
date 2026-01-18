# AI Model Analysis Package
from .scraper import scrape_paperswithcode, scrape_multiple_tasks
from .cleaner import clean_data, load_raw_data, save_cleaned_data
from .analyzer import generate_summary_stats, generate_insights_report
from .models import train_models, evaluate_model, predict_performance
from .utils import setup_logging, validate_data

__version__ = "1.0.0"
