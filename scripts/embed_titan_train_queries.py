"""Create resumable Amazon Titan V2 embeddings for the unique train queries."""

import sys
from pathlib import Path

# Allow direct execution without requiring an editable installation first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from router.config import TRAIN_DATA_PATH, TRAIN_SOURCE_DATA_PATH
from router.embeddings import QueryEmbedder

CHECKPOINT_EVERY = 25


def main() -> None:
    """Embed each unique prompt once and preserve progress after every batch."""
    if not TRAIN_SOURCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Source data not found: {TRAIN_SOURCE_DATA_PATH}")

    source = (
        pd.read_parquet(TRAIN_SOURCE_DATA_PATH)[["query_id", "query", "domain"]]
        .drop_duplicates("query_id")
        .reset_index(drop=True)
    )
    completed = pd.DataFrame(columns=["query_id", "query", "domain", "embedding"])
    if TRAIN_DATA_PATH.exists():
        completed = pd.read_parquet(TRAIN_DATA_PATH).drop_duplicates("query_id")
        print(f"Resuming: {len(completed):,} Titan embeddings already saved.", flush=True)

    done_ids = set(completed["query_id"])
    pending = source[~source["query_id"].isin(done_ids)]
    if pending.empty:
        print(f"Titan embedding file is complete: {TRAIN_DATA_PATH}", flush=True)
        return

    embedder = QueryEmbedder()
    rows = completed.to_dict("records")
    total = len(source)
    for position, row in enumerate(pending.itertuples(index=False), start=len(rows) + 1):
        rows.append({
            "query_id": row.query_id,
            "query": row.query,
            "domain": row.domain,
            "embedding": embedder.embed(row.query).ravel().tolist(),
        })
        if position % CHECKPOINT_EVERY == 0 or position == total:
            output = pd.DataFrame(rows).sort_values("query_id").reset_index(drop=True)
            TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            output.to_parquet(TRAIN_DATA_PATH, index=False)
            print(f"Saved {position:,}/{total:,} Titan embeddings.", flush=True)

    print(f"Finished: {TRAIN_DATA_PATH}", flush=True)


if __name__ == "__main__":
    main()
