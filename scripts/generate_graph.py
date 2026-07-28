# Add all imports
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
    path = Path(file_path)
    if not path.exists():
        logging.error(f"Pipeline failure: File not found at '{file_path}'")
        raise FileNotFoundError(f"Missing required dataset: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logging.info(f"Successfully loaded '{path.name}' ({len(data)} items).")
            return data
    except json.JSONDecodeError as e:
        logging.error(f"Pipeline failure: Invalid JSON format in '{path}' -> {e}")
        raise


def calculate_ai_similarity_weights(
    courses: list, skills: list, threshold: float = 0.25
) -> list:
    """AI/ML Module: Uses TF-IDF vectorization and Cosine Similarity to detect implicit semantic connections
    between course titles and skill labels."""
    logging.info("Running AI/ML semantic similarity engine...")

    course_texts = [c["name"] for c in courses]
    skill_texts = [s["label"] for s in skills]
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
    """Builds a NetworkX graph with dynamic node sizing and Category Hub clusters."""
    G = nx.Graph()
    skill_weights = {s["id"]: 0 for s in skills}

    # Create Category Hub Nodes
    categories = set(s["category"] for s in skills if "category" in s)
    for cat in categories:
        cat_node_id = f"cat_{cat}"
        G.add_node(
            cat_node_id,
            label=cat.upper(),
            title=f"Category Cluster: {cat}",
            category=cat,
            color=CATEGORY_COLORS.get(cat, "#F59E0B"),
            type="category_hub",
            shape="hexagon",
            size=28,
            font={
                "color": "#FFFFFF",
                "size": 13,
                "strokeWidth": 3,
                "strokeColor": "#111827",
            },
        )

    # Add Skill Nodes and link them to their Category Hub
    for skill in skills:
        G.add_node(
            skill["id"],
            label=skill["label"],
            category=skill["category"],
            color=CATEGORY_COLORS.get(skill["category"], "#9CA3AF"),
            type="skill",
        )

        # Connect Skill to its Category Hub
        cat_node_id = f"cat_{skill['category']}"
        if G.has_node(cat_node_id):
            G.add_edge(
                skill["id"],
                cat_node_id,
                weight=2.0,
                edge_type="category",
                color={"color": "#4B5563", "opacity": 0.5},
            )

    # Add Course Nodes & Explicit Edges
    for course in courses:
        G.add_node(
            course["id"],
            label=course["name"],
            category="Course",
            color=CATEGORY_COLORS.get("Course", "#3B82F6"),
            type="course",
            size=10,
        )
        for skill_id in course.get("skills", []):
            if skill_id in skill_weights:
                skill_weights[skill_id] += 1
                G.add_edge(course["id"], skill_id, weight=1.0, edge_type="explicit")
            else:
                logging.warning(
                    f"Validation Warning: Skill ID '{skill_id}' in course '{course['id']}' not found in skills.json"
                )

    # Add AI Implicit Edges
    for link in ai_links:
        s_id = link["skill_id"]
        c_id = link["course_id"]
        if G.has_node(s_id) and G.has_node(c_id):
            if not G.has_edge(c_id, s_id):
                G.add_edge(c_id, s_id, weight=link["score"], edge_type="implicit")
                skill_weights[s_id] += 0.5
                logging.info(f"Added AI implicit edge: {c_id} <--> {s_id}")

    # Calculate Dynamic Node Sizes for Skills
    for skill_id, weight in skill_weights.items():
        if G.has_node(skill_id):
            calculated_size = 15 + (weight * 8)
            G.nodes[skill_id]["size"] = calculated_size

    return G


def render_interactive_html(G: nx.Graph, output_path: str = "docs/index.html"):
    """Converts the NetworkX graph into an interactive PyVis HTML visualizer with customized physics and styling."""
    logging.info(f"Rendering interactive graph to '{output_path}'...")

    # Ensure target output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize PyVis Network
    net = Network(height="750px", width="100%", bgcolor="#1F2937", font_color="#F3F4F6")

    # Import NetworkX graph structure
    net.from_nx(G)

    # Configure Physics Engine and Visual Behavior
    options_dict = {
        "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "font": {
                "color": "#FFFFFF",
                "size": 16,
                "face": "Tahoma",
                "strokeWidth": 3,
                "strokeColor": "#1F2937",
            },
        },
        "edges": {
            "color": {"color": "#6B7280", "highlight": "#F59E0B", "inherit": False},
            "smooth": {"type": "continuous"},
        },
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -12000,
                "centralGravity": 0.3,
                "springLength": 95,
                "springConstant": 0.04,
                "damping": 0.09,
            },
            "minVelocity": 0.75,
        },
    }

    net.set_options(json.dumps(options_dict))

    # Save output to static HTML
    try:
        net.write_html(str(output_file))
        logging.info(f"Successfully generated interactive graph at: {output_file}")
    except Exception as e:
        logging.error(f"Failed to save HTML graph visualization: {e}")
        raise e


def main():
    logging.info("Starting Knowledge Graph pipeline execution...")
    # 1. Load Data
    skills = load_json_data("data/skills.json")
    courses = load_json_data("data/courses.json")

    if not skills or not courses:
        logging.error("Pipeline aborted: Essential data files are missing or empty.")
        return

    # 2. ML Engine: Calculate Implicit AI Connections
    ai_links = calculate_ai_similarity_weights(courses, skills, threshold=0.50)

    # 3. Build Graph Logic
    G = build_knowledge_graph(skills, courses, ai_links)

    # 4. Render HTML Output
    render_interactive_html(G, output_path="docs/index.html")
    logging.info("Pipeline executed successfully!")


if __name__ == "__main__":
    main()
