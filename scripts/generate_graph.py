import json
import logging
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network


# Configure logging for pipeline integration and clear executing tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s %(message)s]",
    handlers=[logging.StreamHandler()],
)

# Set colors for each skill category
CATEGORY_COLORS = {
    "Programming": "#3B82F6",
    "Databases": "#10B981",
    "CS Concepts": "#8B5CF6",
    "Infrastructure": "#F59E0B",
    "Math": "#EC4899",
    "Business": "#6B7280",
    "Humanities": "#14B8A6",
    "Science": "#84CC16",
    "Course": "#000000",
}


def load_json_data(file_path: Path) -> list:
    """Safely load JSON data with pipeline-ready error handling."""
    if not file_path.exists():
        logging.error(f"Pipeline failure: File not found at '{file_path}'")
        raise FileNotFoundError(f"Missing required dataset: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logging.info(f"Successfully loaded '{file_path.name}' ({len(data)} items).")
            return data
    except json.JSONDecodeError as e:
        logging.error(f"Pipeline failure: Invalid JSON format in '{file_path}' -> {e}")
        raise


def calculate_ai_similarity_weights(
    course: list, skills: list, threshold: float = 0.25
) -> list:
    """AI/ML Module: Uses TF-IDF vectorization and Cosine Similarity to detect implicit semantic connections 
        between course titles and skill labels."""
    logging.info("Running AI/ML semantic similarity engine...")

    course_texts =


# Paths setup
DATA_DIR = Path("data")
skills_file = DATA_DIR / "skills.json"
courses_file = DATA_DIR / "courses.json"

# Load datasets with error handling
try:
    skills_data = load_json_data(skills_file)
    courses_data = load_json_data(courses_file)
except Exception as e:
    logging.critical(f"Graph generation aborted due to data loading failure: {e}")
    exit(1)
