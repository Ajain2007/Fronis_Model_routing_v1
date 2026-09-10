"""Difficulty profile construction and cost-aware model selection."""

import pandas as pd

MODEL_COLUMNS = {
    "gemma-2-9b-it": "model1_pass",
    "llama-3-8b-instruct-lite": "model2_pass",
    "mistral-7b-instruct-v0.3": "model3_pass",
}


def build_routing_profile(difficulty_df: pd.DataFrame) -> pd.DataFrame:
    """Compute each selected model's historical pass rate at every difficulty."""
    rows = []
    for difficulty in ("D1", "D2", "D3", "D4"):
        group = difficulty_df[difficulty_df["difficulty"].eq(difficulty)]
        for model, column in MODEL_COLUMNS.items():
            rows.append({"difficulty": difficulty, "model": model, "success_rate": float(group[column].mean())})
    return pd.DataFrame(rows).pivot(index="difficulty", columns="model", values="success_rate")


class DifficultyRouter:
    def __init__(self, routing_profile: pd.DataFrame, model_cost_order: list[str], min_success_rate: float) -> None:
        self.profile = routing_profile
        self.model_cost_order = model_cost_order
        self.min_success_rate = min_success_rate

    def select_model(self, difficulty: str) -> tuple[str, float]:
        """Return the cheapest model whose observed success rate meets the policy."""
        rates = self.profile.loc[difficulty]
        for model in self.model_cost_order:
            if rates[model] >= self.min_success_rate:
                return model, float(rates[model])
        best = str(rates.idxmax())
        return best, float(rates[best])
