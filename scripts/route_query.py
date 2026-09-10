"""Route one incoming query and print the compact JSON result."""

import argparse
import sys
from pathlib import Path

# Allow `python scripts/route_query.py ...` without an editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json
from getpass import getpass
import os

import joblib
import pandas as pd
from openai import OpenAI
from setfit import SetFitModel

from router.config import (DIFFICULTY_INDEX_PATH, EMBEDDING_MODEL, KNN_PATH, MIN_SUCCESS_RATE,
                           MODEL_COST_ORDER, RERANKER_MODEL_NAME, ROUTING_PROFILE_PATH, SETFIT_PATH)
from router.difficulty import DifficultyRouter
from router.domain_classifier import DomainClassifier
from router.embeddings import QueryEmbedder
from router.pipeline import ModelRoutingPipeline
from router.reranker import QueryReranker
from router.retrieval import DomainRetriever


def build_pipeline() -> ModelRoutingPipeline:
    """Load saved artifacts once and compose the production pipeline."""
    required = (KNN_PATH, SETFIT_PATH, DIFFICULTY_INDEX_PATH, ROUTING_PROFILE_PATH)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Run scripts/build_router_artifacts.py first. Missing: " + ", ".join(missing))
    client = OpenAI()
    return ModelRoutingPipeline(
        QueryEmbedder(client, EMBEDDING_MODEL),
        DomainClassifier(joblib.load(KNN_PATH), SetFitModel.from_pretrained(str(SETFIT_PATH))),
        DomainRetriever(pd.read_parquet(DIFFICULTY_INDEX_PATH)),
        QueryReranker(RERANKER_MODEL_NAME),
        DifficultyRouter(pd.read_parquet(ROUTING_PROFILE_PATH), MODEL_COST_ORDER, MIN_SUCCESS_RATE),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Route a query to a model.")
    parser.add_argument("query", nargs="?", help="Query text; prompts when omitted.")
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key: ")
    query = (args.query or input("Enter query: ")).strip()
    print(json.dumps(build_pipeline().route(query), indent=2))


if __name__ == "__main__":
    main()
