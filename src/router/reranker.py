"""Cross-encoder reranking for retrieved historical queries."""

import pandas as pd
from sentence_transformers import CrossEncoder


class QueryReranker:
    def __init__(self, model_name: str) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return candidates
        pairs = [(query, text) for text in candidates["query"]]
        result = candidates.copy()
        result["reranker_score"] = self.model.predict(pairs)
        return result.sort_values("reranker_score", ascending=False).reset_index(drop=True)
