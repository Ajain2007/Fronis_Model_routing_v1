# Fronis Router

Production query router using only the previously trained SetFit model and a
saved KNN classifier. It classifies an input into the top two domains, retrieves
and reranks historical traces within each domain, estimates the required
difficulty, and selects the cheapest qualifying model.

## Setup

Use an isolated virtual environment. Do not install this project into the
global Python environment because other projects may require incompatible
TensorFlow/Hugging Face versions.

```powershell
cd fronis_updated
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/build_router_artifacts.py
$env:OPENAI_API_KEY = "your_api_key"
python scripts/route_query.py "Write Python code to calculate the probability of drawing two aces"
```

`build_router_artifacts.py` creates KNN and retrieval artifacts only. It never
re-trains or changes `models/setfit`, which contains the reused trained encoder
and fitted classification head.

Example output:

```json
{
  "query": "...",
  "domain": "Coding",
  "best_match": "train-2145",
  "difficulty": "D3",
  "model": "gemma-2-9b-it",
  "expected_success_rate": 1.0
}
```
