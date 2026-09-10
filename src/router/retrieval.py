"""Domain-scoped semantic retrieval over labeled historical difficulty traces."""

import numpy as np
import pandas as pd

from router.embeddings import normalize


class DomainRetriever:
    def __init__(self, difficulty_df: pd.DataFrame) -> None:
        self.df = difficulty_df.reset_index(drop=True).copy()
        self.vectors = normalize(np.asarray(self.df["embedding"].tolist(), dtype=np.float32))
        self.domain_positions = {
            domain: np.flatnonzero(self.df["domain"].eq(domain).to_numpy())
            for domain in self.df["domain"].unique()
        }

    def retrieve(self, query_embedding: np.ndarray, domain: str, top_k: int) -> pd.DataFrame:
        positions = self.domain_positions.get(domain)
        if positions is None:
            return pd.DataFrame()
        similarities = (query_embedding @ self.vectors[positions].T).ravel()
        best = np.argsort(similarities)[::-1][:top_k]
        result = self.df.iloc[positions[best]].copy()
        result["similarity"] = similarities[best]
        return result.sort_values("similarity", ascending=False).reset_index(drop=True)
