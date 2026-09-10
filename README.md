# Fronis Model Routing

Fronis Model Routing selects a suitable LLM for an incoming query using historical xRouteBench routing traces. The application does **not** retrain a model while serving a query.

## What the router does

1. Creates an embedding for the incoming query.
2. Uses the saved KNN and saved SetFit classifier to identify two candidate domains: `Coding`, `Knowledge`, `Math`, or `Reasoning / Logic`.
3. Searches only the historical traces in each candidate domain.
4. Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank the five retrieved traces per domain.
5. Combines normalized embedding similarity (30%) and cross-encoder relevance (70%) to choose one final domain and historical best match.
6. Uses that match's difficulty (`D1`–`D4`) and historical model pass rates to recommend the lowest-cost suitable model.

Example result:

```json
{
  "query": "Write Python code to calculate the probability of drawing two aces",
  "domain": "Coding",
  "best_match": "train-2145",
  "difficulty": "D3",
  "model": "gemma-2-9b-it",
  "expected_success_rate": 1.0
}
```

## Repository layout

```text
data/processed/                    Prepared benchmark data and query embeddings
models/setfit/                     Preserved, exported SetFit model weights
models/knn/                        Generated KNN serving artifacts
models/routing/                    Generated similarity index and pass-rate profile
scripts/build_router_artifacts.py  Builds KNN and retrieval artifacts
scripts/route_query.py             Routes one query from the command line
src/router/                        Production routing implementation
```

## One-time setup

Run these commands in PowerShell from the repository root:

```powershell
cd "C:\path\to\Fronis_Model_routing_v1"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Always activate the virtual environment before running the project:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Build the local artifacts

The first build creates the KNN model and retrieval files:

```powershell
python scripts\build_router_artifacts.py
```

It creates:

```text
models/knn/knn_classifier.joblib
models/knn/training_queries.parquet
models/routing/difficulty_index.parquet
models/routing/routing_profile.parquet
```

This command **does not retrain or modify SetFit**. The saved SetFit weights and classification head are preserved in `models/setfit/`.

Run this build only after a fresh clone, or when the historical embeddings or benchmark data change. Do not run it for every incoming query.

## Route a query

The current checked-in implementation uses the OpenAI embedding client. Set a valid key only in the current terminal session, then route a query:

```powershell
$env:OPENAI_API_KEY = "your_openai_key"
python scripts\route_query.py "Write Python code to calculate the probability of drawing two aces"
```

The first invocation may download the cross-encoder model into the local Hugging Face cache. Later invocations reuse the cached model and the saved KNN/SetFit artifacts.

## Difficulty and recommended-model policy

The prepared dataset assigns difficulty from historical pass/fail outcomes:

| Pattern | Gemma | Llama | Mistral | Difficulty |
|---|---:|---:|---:|---|
| `111` | pass | pass | pass | D1 |
| `110` | pass | pass | fail | D2 |
| `100` | pass | fail | fail | D3 |
| `000` | fail | fail | fail | D4 |

For the selected difficulty, the router chooses the least expensive model that meets the configured minimum historical success rate. Configuration is in `src/router/config.py`; selection logic is in `src/router/difficulty.py`.

## Amazon Bedrock Titan migration

AWS SSO access to Titan V2 has been verified for the configured profile. The current source code still uses OpenAI embeddings. Before changing the provider, all historical vectors and the KNN/retrieval artifacts must be regenerated in the same Titan vector space; mixing Titan query embeddings with the existing OpenAI vectors is invalid.

After the Titan implementation is added, authenticate and set the AWS profile in each new terminal:

```powershell
aws sso login --profile ai-runtime-governor-729297430338
$env:AWS_PROFILE = "ai-runtime-governor-729297430338"
$env:AWS_REGION = "us-east-1"
```

The role must permit `bedrock:InvokeModel` for `amazon.titan-embed-text-v2:0`.

## Troubleshooting

**`ModuleNotFoundError: No module named 'router'`**  
Run commands from the repository root. The scripts also add `src/` to Python's path, so an editable installation is not required solely for this import.

**SetFit / transformers import errors**  
Activate `.venv` and install dependencies with `python -m pip install -e .`. Avoid installing the project into global Python.

**AWS CLI is not recognized**  
Restart VS Code or PowerShell after installing AWS CLI. It is available at `C:\Program Files\Amazon\AWSCLIV2\aws.exe` until the refreshed `PATH` is picked up.

**GitHub push returns 403**  
Accept the repository collaborator invitation using the GitHub identity that will push the code, then retry `git push -u origin main`.
