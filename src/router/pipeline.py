"""End-to-end query routing pipeline."""

from router.config import DIFFICULTY_VALUE, EMBEDDING_WEIGHT, RERANKER_WEIGHT, SIMILAR_QUERIES_PER_DOMAIN, TOP_N_DOMAINS


class ModelRoutingPipeline:
    def __init__(self, embedder, classifier, retriever, reranker, difficulty_router) -> None:
        self.embedder = embedder
        self.classifier = classifier
        self.retriever = retriever
        self.reranker = reranker
        self.difficulty_router = difficulty_router

    def route(self, query: str) -> dict:
        """Select domains, match historic traces, infer difficulty, and choose a model."""
        embedding = self.embedder.embed(query)
        classification = self.classifier.classify(query, embedding, top_n=TOP_N_DOMAINS)
        matches = {}

        for domain in classification["domains"]:
            candidates = self.retriever.retrieve(embedding, domain, top_k=SIMILAR_QUERIES_PER_DOMAIN)
            reranked = self.reranker.rerank(query, candidates)
            if not reranked.empty:
                matches[domain] = reranked.iloc[0].to_dict()

        self._add_match_scores(matches)
        if not matches:
            raise RuntimeError("No historical matches found for the selected domains.")

        # Select one final domain using the fused match score.  Difficulty is
        # deliberately only a tie-breaker: a distant D4 trace must not override
        # a substantially better Math/Coding match merely because it is harder.
        final_domain, final_match = max(
            matches.items(),
            key=lambda item: (item[1]["final_match_score"], DIFFICULTY_VALUE[item[1]["difficulty"]]),
        )
        model, success_rate = self.difficulty_router.select_model(final_match["difficulty"])
        return {
            "query": query,
            "domain": final_domain,
            "best_match": str(final_match["query_id"]),
            "difficulty": str(final_match["difficulty"]),
            "model": model,
            "expected_success_rate": success_rate,
        }

    @staticmethod
    def _add_match_scores(matches: dict) -> None:
        """Fuse normalized embedding and reranker scores across selected domains."""
        similarities = [match["similarity"] for match in matches.values()]
        reranker_scores = [match["reranker_score"] for match in matches.values()]
        sim_min, sim_max = min(similarities), max(similarities)
        rank_min, rank_max = min(reranker_scores), max(reranker_scores)
        for match in matches.values():
            sim = 1.0 if sim_min == sim_max else (match["similarity"] - sim_min) / (sim_max - sim_min)
            rerank = 1.0 if rank_min == rank_max else (match["reranker_score"] - rank_min) / (rank_max - rank_min)
            match["final_match_score"] = EMBEDDING_WEIGHT * sim + RERANKER_WEIGHT * rerank
