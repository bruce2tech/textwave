"""assignment-9-test/unit-test.py

Unit tests for Assignment 9

Run with:
    cd assignment-9-test/
    PYTHONPATH="../textwave" python -m unittest unit-test.py
"""

import unittest
from typing import List
from typing import Any
from types import ModuleType
from unittest.mock import patch
import json
import socket
import subprocess
import time
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Iterator

import requests

import numpy as np
import torch

from modules.retrieval.reranker import Reranker  # Make sure your class sits in reranker.py

# ---------------------------------------------------------------------------
# Lightweight stubs for the HuggingFace model + tokenizer
# ---------------------------------------------------------------------------

class _DummyModel:  # mimics AutoModelForSequenceClassification
    """Returns pre-configured logits so we can assert ranking order."""

    def __init__(self, scores: List[float]):
        self._scores = scores  # length will be set dynamically from test

    def __call__(self, **kwargs):  # pylint: disable=unused-argument
        class _Out:  # simple container for .logits
            pass

        out = _Out()
        # Ensure logits shape (N, 1) where N = len(self._scores)
        out.logits = torch.tensor(self._scores, dtype=torch.float32).unsqueeze(1)
        return out


class _DummyTokenizer:  # mimics AutoTokenizer
    """Produces a minimal tensor dict accepted by the dummy model."""

    def __call__(self, pairs, padding=True, truncation=True, return_tensors="pt"):  # type: ignore[override]  # pylint: disable=unused-argument
        n = len(pairs)
        max_len = 4  # arbitrary
        return {
            "input_ids": torch.zeros((n, max_len), dtype=torch.long),
            "attention_mask": torch.ones((n, max_len), dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Helper: factory that yields a *patched* Reranker instance
# ---------------------------------------------------------------------------

def _make_mock_reranker(rtype: str, mock_scores: List[float] | None = None) -> Reranker:
    """Create a Reranker whose cross-encoder parts are stubbed out."""
    mock_scores = mock_scores or []

    # Temporarily patch the HF load calls
    with patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=_DummyModel(mock_scores)):
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=_DummyTokenizer()):
            rer = Reranker(type=rtype)
    # We return the instance *outside* the context manager so the stubs live on
    return rer


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestReranker(unittest.TestCase):
    """Tests each reranking strategy in isolation with deterministic inputs."""

    # Common corpus for lexical-based tests
    _RETRIEVED_DOCS: List[str] = [
        "apple pie recipe",
        "banana bread instructions",
        "bake delicious apple turnover",
        "fresh orange juice benefits",
    ]
    _QUERY = "apple dessert"

    # --------------------------------------------------------------
    # cross-encoder
    # --------------------------------------------------------------

    def test_cross_encoder_rerank_orders_by_logits(self):
        """Docs should be sorted descending by dummy logits produced by the model."""
        dummy_scores = [0.1, 0.9, 0.4, 0.2]  # pre-baked relevance
        rer = _make_mock_reranker("cross_encoder", dummy_scores)

        ranked_docs, indices, scores = rer.rerank(self._QUERY, self._RETRIEVED_DOCS)

        # Expect ordering matching descending scores
        expected_order = np.argsort(dummy_scores)[::-1]  # indices 1,2,3,0
        self.assertEqual(indices, expected_order.tolist())
        self.assertEqual(ranked_docs, [self._RETRIEVED_DOCS[i] for i in expected_order])
        self.assertEqual(scores, [dummy_scores[i] for i in expected_order])

    # --------------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------------

    def test_tfidf_rerank_prefers_apple_docs(self):
        """TF-IDF distance should rank apple-related docs highest (lowest distance)."""
        rer = _make_mock_reranker("tfidf")
        ranked_docs, indices, distances = rer.rerank(self._QUERY, self._RETRIEVED_DOCS, distance_metric="cosine")

        # First result should mention "apple"
        self.assertIn("apple", ranked_docs[0])
        # Ensure list lengths and index mapping are correct
        self.assertEqual(len(ranked_docs), len(self._RETRIEVED_DOCS))
        self.assertEqual(indices[0], self._RETRIEVED_DOCS.index(ranked_docs[0]))
        # Distances array should be ascending (most similar first)
        self.assertTrue(all(d1 <= d2 for d1, d2 in zip(distances, distances[1:])))

    # --------------------------------------------------------------
    # Bag-of-Words
    # --------------------------------------------------------------

    def test_bow_rerank_prefers_apple_docs(self):
        """BoW similarity (e.g., cosine on counts) should also rank apple docs first."""
        rer = _make_mock_reranker("bow")
        ranked_docs, indices, distances = rer.rerank(self._QUERY, self._RETRIEVED_DOCS, distance_metric="cosine")

        self.assertIn("apple", ranked_docs[0])
        self.assertEqual(len(ranked_docs), len(self._RETRIEVED_DOCS))
        # Verify ordering ascending by distance
        self.assertTrue(all(d1 <= d2 for d1, d2 in zip(distances, distances[1:])))

    # --------------------------------------------------------------
    # Sequential (TF-IDF -> Cross-encoder)
    # --------------------------------------------------------------

    def test_sequential_rerank_two_phase_pipeline(self):
        """Sequential strategy should first filter with TF-IDF then reorder with logits."""
        # Step 1: Create logits that deliberately invert the TF-IDF top-2 order
        # We *expect* TF-IDF to pick the two apple docs (idx 0 and 2). We'll
        # give higher cross-encoder score to doc idx 2 so the final ranking
        # flips them.
        dummy_scores = [0.2, 0.1, 0.9, 0.05]
        rer = _make_mock_reranker("sequential", dummy_scores)

        seq_k1, seq_k2 = 2, 2
        ranked_docs, indices, final_scores = rer.rerank(self._QUERY, self._RETRIEVED_DOCS, seq_k1=seq_k1, seq_k2=seq_k2)

        # Ensure we received exactly k2 results
        self.assertEqual(len(ranked_docs), seq_k2)
        # Both results should come from the TF-IDF top-k1 subset (apple docs)
        self.assertTrue(all("apple" in doc for doc in ranked_docs))
        # Cross-encoder score should place doc with higher dummy_scores[2] on top
        self.assertEqual(ranked_docs[0], self._RETRIEVED_DOCS[2])
        # Sanity: scores descending
        self.assertTrue(all(s1 >= s2 for s1, s2 in zip(final_scores, final_scores[1:])))


"""
Unit-test suite for the Flask QA service boilerplate.

It purposely *fails* the two methods are completed:
    * ``initialize_index`` - must build & return a FAISS index.
    * ``generate_answer`` - must implement the retrieval-and-generate
      pipeline and populate ``answer`` in the JSON response.

The suite uses Flask's built-in test-client to avoid spawning a server.
Edge-case coverage includes:
    * Empty corpus / uninitialised index
    * Missing / empty JSON payload
    * Unicode & long-string queries
"""

MODULE_NAME = "app"


class TestFlaskQAService(unittest.TestCase):
    """Comprehensive edge-case tests for the Flask QA service."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod: ModuleType = import_module(MODULE_NAME)

        # Build Flask test client
        cls.app = cls.mod.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    # ------------------------------------------------------------------
    # initialize_index tests
    # ------------------------------------------------------------------
    def test_initialize_index_returns_object(self) -> None:
        """initialize_index should not return None once implemented."""
        index = self.mod.initialize_index()
        self.assertIsNotNone(index, "initialize_index() returned None - implement the index builder!")

    # ------------------------------------------------------------------
    # /generate endpoint tests
    # ------------------------------------------------------------------
    def _post_generate(self, payload: Any) -> tuple[int, dict[str, Any]]:
        """Helper to POST /generate and return (status_code, json)."""
        resp = self.client.post(
            "/generate",
            data=json.dumps(payload),
            content_type="application/json",
        )
        return resp.status_code, resp.get_json(force=True, silent=True) or {}

    def test_generate_success_basic(self) -> None:
        """Valid query should yield JSON with non-empty 'answer'."""
        status, data = self._post_generate({"query": "Who is Albert Einstein?"})
        self.assertEqual(status, 200)
        self.assertIn("answer", data)
        # Will be None until implemented
        self.assertIsNotNone(data["answer"], "/generate returned 'answer': null - implement the pipeline!")

    def test_generate_missing_json(self) -> None:
        """POST without JSON payload should return 400 Bad Request."""
        resp = self.client.post("/generate")
        self.assertEqual(resp.status_code, 400, "/generate without JSON must return 400")

    def test_generate_empty_query(self) -> None:
        """Empty query string should yield 422 or meaningful error."""
        status, _ = self._post_generate({"query": "   "})
        self.assertIn(status, {400, 422}, "Empty query should be rejected with 400/422")

    def test_generate_unicode_query(self) -> None:
        """Unicode characters should be processed without error."""
        status, data = self._post_generate({"query": "¿Cuál es la teoría de la relatividad?"})
        self.assertEqual(status, 200)
        self.assertIn("answer", data)

    def test_generate_long_query(self) -> None:
        """Very long query should not crash the service."""
        long_query = "What is " + ("very " * 1000) + "long query?"
        status, _ = self._post_generate({"query": long_query})
        self.assertEqual(status, 200)


"""
End-to-end tests for the Flask-based TextWave micro-service (textwave/app.py) running
inside a Docker container.

Steps:
------------
1.  Builds the Docker image (tag: textwave:test).
2.  Starts a container on a random free host port.
3.  POSTs a sample query to `/generate`.
4.  Asserts HTTP 200 and that the JSON payload contains a non-empty
    "answer" string.
5.  Cleans up the container and image afterward.
"""

# ---------------------------------------------------------------------------#
# Helper utilities                                                            #
# ---------------------------------------------------------------------------#
def _find_free_port() -> int:
    """Return an unused TCP port on the host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@contextmanager
def _docker_container(image: str, host_port: int) -> Iterator[str]:
    """
    Run *image* in detached mode, mapping container port 5000 → host_port.
    Yields the container ID and guarantees cleanup.
    """
    run_cmd = [
        "docker",
        "run",
        "-d",
        "-p",
        f"{host_port}:5000",
        "--rm",          # auto-remove when container exits
        image,
    ]
    container_id = subprocess.check_output(run_cmd, text=True).strip()
    try:
        yield container_id
    finally:
        subprocess.call(
            ["docker", "kill", container_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class TestDockerService(unittest.TestCase):
    """End-to-end tests against the running Dockerised QA service."""

    DOCKERFILE_PATH = Path("Dockerfile")
    IMAGE_TAG = "textwave:test"
    BUILD_TIMEOUT_SEC = 600
    CONTAINER_TIMEOUT_SEC = 25
    POLL_INTERVAL_SEC = 1.0

    # -- class lifecycle --------------------------------------------------
    @classmethod
    def setUpClass(cls):
        if not cls.DOCKERFILE_PATH.exists():
            raise unittest.SkipTest(f"No Dockerfile found at {cls.DOCKERFILE_PATH}")

        build_cmd = [
            "docker",
            "build",
            "-t",
            cls.IMAGE_TAG,
            ".",
        ]
        result = subprocess.run(
            build_cmd,
            capture_output=True,
            text=True,
            timeout=cls.BUILD_TIMEOUT_SEC,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Docker build failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

    @classmethod
    def tearDownClass(cls):
        subprocess.call(
            ["docker", "rmi", "-f", cls.IMAGE_TAG],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # -- actual test ------------------------------------------------------
    def test_generate_returns_nonempty_answer(self):
        """POST /generate should return 200 and a non-empty answer."""
        host_port = _find_free_port()
        base_url = f"http://127.0.0.1:{host_port}"

        with _docker_container(self.IMAGE_TAG, host_port):
            # Wait until the container starts accepting connections
            deadline = time.time() + self.CONTAINER_TIMEOUT_SEC
            healthy = False
            while time.time() < deadline and not healthy:
                try:
                    requests.get(base_url + "/")  # root may 404 but proves server is up
                    healthy = True
                except requests.ConnectionError:
                    time.sleep(self.POLL_INTERVAL_SEC)

            self.assertTrue(healthy, "The container never became reachable.")

            payload = {
                "query": "What is the role of antioxidants in green tea?"
            }
            resp = requests.post(
                base_url + "/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=10,
            )

            self.assertEqual(
                resp.status_code, 200, f"Expected HTTP 200, got {resp.status_code}"
            )

            data = resp.json()
            self.assertIn("answer", data, 'JSON must contain key "answer"')
            self.assertIsInstance(data["answer"], str)
            self.assertTrue(data["answer"].strip(), "Answer must be non-empty string")

            self.assertIn("query", data, 'JSON must contain key "query"')
            self.assertIsInstance(data["query"], str)
            self.assertTrue(data["query"].strip(), "Query must be non-empty string")


if __name__ == "__main__":
    unittest.main(verbosity=2)
