"""Amazon Bedrock Titan V2 embedding adapter."""

import json
import os
import time

import boto3
import numpy as np
from botocore.exceptions import ClientError

from router.config import TITAN_DIMENSIONS, TITAN_MODEL_ID


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Return L2-normalized vectors for cosine similarity/distance operations."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class QueryEmbedder:
    """Create a Titan V2 float embedding through the configured AWS profile."""

    def __init__(self, client=None, model_name: str = TITAN_MODEL_ID, dimensions: int = TITAN_DIMENSIONS) -> None:
        profile = os.environ.get("AWS_PROFILE")
        region = os.environ.get("AWS_REGION")
        session = boto3.Session(profile_name=profile, region_name=region) if profile else boto3.Session(region_name=region)
        self.client = client or session.client("bedrock-runtime")
        self.model_name = model_name
        self.dimensions = dimensions

    def embed(self, query: str) -> np.ndarray:
        """Embed one non-empty query, retrying temporary Bedrock failures."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        if len(query) > 50_000:
            raise ValueError("Titan V2 accepts at most 50,000 input characters.")

        request = json.dumps({
            "inputText": query,
            "dimensions": self.dimensions,
            "normalize": True,
            "embeddingTypes": ["float"],
        })
        transient_errors = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException"}
        for attempt in range(6):
            try:
                response = self.client.invoke_model(
                    modelId=self.model_name,
                    body=request,
                    contentType="application/json",
                    accept="application/json",
                )
                payload = json.loads(response["body"].read())
                vector = np.asarray(payload["embedding"], dtype=np.float32).reshape(1, -1)
                return normalize(vector)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in transient_errors or attempt == 5:
                    raise RuntimeError(f"Titan embedding failed ({code}): {exc}") from exc
                time.sleep(min(2 ** attempt, 20))

        raise RuntimeError("Titan embedding retry loop ended unexpectedly.")
