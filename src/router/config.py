"""Paths and tunable policy constants for the routing pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_DATA_PATH = DATA_DIR / "fronis_router_train_embedded.parquet"
DIFFICULTY_DATA_PATH = DATA_DIR / "fronis_difficulty_dataset.parquet"
KNN_PATH = MODEL_DIR / "knn" / "knn_classifier.joblib"
KNN_EXAMPLES_PATH = MODEL_DIR / "knn" / "training_queries.parquet"
SETFIT_PATH = MODEL_DIR / "setfit"
DIFFICULTY_INDEX_PATH = MODEL_DIR / "routing" / "difficulty_index.parquet"
ROUTING_PROFILE_PATH = MODEL_DIR / "routing" / "routing_profile.parquet"

EMBEDDING_MODEL = "text-embedding-3-small"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_N_DOMAINS = 2
KNN_NEIGHBORS = 11
SIMILAR_QUERIES_PER_DOMAIN = 5
EMBEDDING_WEIGHT = 0.30
RERANKER_WEIGHT = 0.70
MIN_SUCCESS_RATE = 0.80
MODEL_COST_ORDER = ["mistral-7b-instruct-v0.3", "llama-3-8b-instruct-lite", "gemma-2-9b-it"]
DIFFICULTY_VALUE = {"D1": 1, "D2": 2, "D3": 3, "D4": 4}
