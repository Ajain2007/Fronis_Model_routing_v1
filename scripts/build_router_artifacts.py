"""Build KNN and retrieval artifacts; never retrains or modifies SetFit."""

import sys
from pathlib import Path

# Allow `python scripts/build_router_artifacts.py` without an editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from router.config import (DIFFICULTY_DATA_PATH, DIFFICULTY_INDEX_PATH, KNN_EXAMPLES_PATH,
                           KNN_NEIGHBORS, KNN_PATH, ROUTING_PROFILE_PATH, TRAIN_DATA_PATH)
from router.difficulty import build_routing_profile
from router.embeddings import normalize


def main() -> None:
    train = pd.read_parquet(TRAIN_DATA_PATH).drop_duplicates("query_id").reset_index(drop=True)
    vectors = normalize(np.asarray(train["embedding"].tolist(), dtype=np.float32))
    knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric="cosine", weights="distance", n_jobs=-1)
    knn.fit(vectors, train["domain"])
    KNN_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(knn, KNN_PATH)
    train[["query_id", "query", "domain"]].to_parquet(KNN_EXAMPLES_PATH, index=False)

    difficulty = pd.read_parquet(DIFFICULTY_DATA_PATH)
    lookup = train.set_index("query_id")["embedding"]
    difficulty["embedding"] = difficulty["query_id"].map(lookup)
    if difficulty["embedding"].isna().any():
        raise RuntimeError("Some difficulty records do not have a train embedding.")
    DIFFICULTY_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    difficulty.to_parquet(DIFFICULTY_INDEX_PATH, index=False)
    build_routing_profile(difficulty).to_parquet(ROUTING_PROFILE_PATH)
    print("Built KNN, difficulty retrieval index, and routing profile. SetFit was untouched.")


if __name__ == "__main__":
    main()
