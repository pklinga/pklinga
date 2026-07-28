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
    courses: list, skills: list, threshold: float = 0.25
) -> list:
    """AI/ML Module: Uses TF-IDF vectorization and Cosine Similarity to detect implicit semantic connections
    between course titles and skill labels."""
    logging.info("Running AI/ML semantic similarity engine...")

    course_texts = [c["name"] for c in courses]
    skill_texts = [s["name"] for s in skills]
    all_texts = course_texts + skill_texts

    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        course_vectors = tfidf_matrix[: len(courses)]
        skill_vectors = tfidf_matrix[len(courses) :]

        similarity_matrix = cosine_similarity(course_vectors, skill_vectors)

        ai_connections = []
        for c_idx, course in enumerate(courses):
            for s_idx, skill in enumerate(skills):
                score = similarity_matrix[c_idx, s_idx]
                if score >= threshold:
                    ai_connections.append(
                        {
                            "course_id": course["id"],
                            "skill_id": skill["id"],
                            "score": float(score),
                        }
                    )
                    logging.info(
                        f"AI Detected implicit connection: '{course['name']}' -> '{skill['label']}' (Score: {score:.2f})"
                    )
        return ai_connections
    except Exception as e:
        logging.warning(
            f"AI Similarity engine encountered an issue, skipping implicit links: {e}"
        )
        return []


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


def build_knowledge_graph(skills: list, courses: list, ai_links: list) -> nx.Graph:
    """Builds a NetworkX graph with dynamic node sizing based on connections."""
    G = nx.Graph()
    skill_weights = {s["id"]: 0 for s in skills}
    for skill in skills:
        G.add_node(
            skill["id"],
            label=skill["label"],
            category=skill["category"],
            color=CATEGORY_COLORS.get(skill["category"], "#9CA3AF"),
            type="skill",
        )
    for course in courses:
        for skill_id in course.get("skills", []):
            if skill_id in skill_weights:
                skill_weights[skill_id] += 1
                G.add_edge(course["id"], skill_id, weight=1.0, edge_type="explicit")
            else:
                logging.warning(
                    f"Validation Warning: Skill ID '{skill_id}' in course '{course['id']}' not found in skills.json"
                )
        G.add_node(
            course["id"],
            label=course["name"],
            category="Course",
            color=CATEGORY_COLORS["Course"],
            type="course",
            size=10,
        )
    for link in ai_links:
        s_id = link["skill_id"]
        c_id = link["course_id"]
        if G.has_node(s_id) and G.has_node(c_id):
            if not G.has_edge(c_id, s_id):
                G.add_edge(c_id, s_id, weight=link["score"], edge_type="implicit")
                skill_weights[s_id] += 0.5
                logging.info(f"Added AI implicit edge: {c_id} <--> {s_id}")

    for skill_id, weight in skill_weights.items():
        if G.has_node(skill_id):
            calculated_size = 15 + (weight * 8)
            G.nodes[skill_id]["size"] = calculated_size
    return G
