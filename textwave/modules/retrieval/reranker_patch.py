from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Reranker:
    """
    Supports four strategies:
      - 'tfidf'         : TF-IDF vectorization + cosine *distance* (ascending)
      - 'bow'           : Bag-of-words counts + cosine *distance* (ascending)
      - 'cross_encoder' : Cross-encoder logits (descending)
      - 'sequential'    : TF-IDF top-k1 filter, then cross-encoder re-order to k2

    Public API returns: (ranked_docs, ranked_indices, ranked_values)
    where ranked_values are distances (for tfidf/bow) or scores (for CE/sequential).
    """

    def __init__(self, type: str = "tfidf", model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.type = type
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

    # ---------------------------- public API ----------------------------

    def rerank(
        self,
        query: str,
        documents: List[str],
        *,
        k: Optional[int] = None,
        seq_k1: int = 4,
        seq_k2: int = 2,
        distance_metric: Optional[str] = None,
    ) -> Tuple[List[str], List[int], List[float]]:
        """
        Dispatch reranking by configured type.
        """
        if self.type == "tfidf":
            return self.tfidf_rerank(query, documents, k=k or len(documents), distance_metric=distance_metric)
        if self.type == "bow":
            return self.bow_rerank(query, documents, k=k or len(documents), distance_metric=distance_metric)
        if self.type == "cross_encoder":
            return self.cross_encoder_rerank(query, documents, k=k or len(documents))
        if self.type == "sequential":
            return self.sequential_rerank(query, documents, seq_k1, seq_k2, distance_metric=distance_metric)

        # Fallback: identity ordering
        return list(documents), list(range(len(documents))), [0.0] * len(documents)

    # ---------------------------- helpers ----------------------------

    @staticmethod
    def _cosine_distance(vec_q, vec_docs) -> np.ndarray:
        """1 - cosine similarity => lower is better (distance)."""
        sims = cosine_similarity(vec_q, vec_docs).ravel()
        return 1.0 - sims

    # ---------------------------- concrete rerankers ----------------------------

    def tfidf_rerank(
        self,
        query: str,
        documents: List[str],
        k: int,
        *,
        distance_metric: Optional[str] = None,
        **_: object,
    ) -> Tuple[List[str], List[int], List[float]]:
        """
        Rank by TF-IDF cosine *distance* (ascending). Return top-k.
        """
        if not documents:
            return [], [], []

        vec = TfidfVectorizer()
        X_docs = vec.fit_transform(documents)
        X_q = vec.transform([query])
        distances = self._cosine_distance(X_q, X_docs)

        order = np.argsort(distances)  # ascending distance
        top = order[:k]
        ranked_docs = [documents[i] for i in top]
        ranked_distances = [float(distances[i]) for i in top]
        return ranked_docs, list(map(int, top)), ranked_distances

    def bow_rerank(
        self,
        query: str,
        documents: List[str],
        k: int,
        *,
        distance_metric: Optional[str] = None,
        **_: object,
    ) -> Tuple[List[str], List[int], List[float]]:
        """
        Rank by Bag-of-Words cosine *distance* (ascending). Return top-k.
        """
        if not documents:
            return [], [], []

        vec = CountVectorizer()
        X_docs = vec.fit_transform(documents)
        X_q = vec.transform([query])
        distances = self._cosine_distance(X_q, X_docs)

        order = np.argsort(distances)  # ascending distance
        top = order[:k]
        ranked_docs = [documents[i] for i in top]
        ranked_distances = [float(distances[i]) for i in top]
        return ranked_docs, list(map(int, top)), ranked_distances

    def _ensure_ce(self) -> None:
        """Lazy-load tokenizer/model. Unit tests patch these constructors with dummies."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def cross_encoder_rerank(
        self,
        query: str,
        documents: List[str],
        k: int,
        **_: object,
    ) -> Tuple[List[str], List[int], List[float]]:
        """
        Use a cross-encoder to score (query, doc) pairs.
        Sort by DESCENDING logits. Return top-k.
        """
        if not documents:
            return [], [], []

        self._ensure_ce()

        pairs = [(query, doc) for doc in documents]
        enc = self._tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")  # type: ignore
        with torch.no_grad():
            out = self._model(**enc)  # type: ignore
        logits = out.logits.squeeze(-1).detach().cpu().numpy().astype(float)

        order = np.argsort(-logits)  # descending score
        top = order[:k]
        ranked_docs = [documents[i] for i in top]
        ranked_scores = [float(logits[i]) for i in top]
        return ranked_docs, list(map(int, top)), ranked_scores

    def sequential_rerank(
        self,
        query: str,
        documents: List[str],
        k1: int,
        k2: int,
        *,
        distance_metric: Optional[str] = None,  # accepted for API compatibility; unused
        **_: object,
    ) -> Tuple[List[str], List[int], List[float]]:
        """
        Two-phase pipeline:
          1) TF-IDF filter to top-k1 (ascending distance)
          2) Cross-encoder re-rank those to top-k2 (descending logits)
          3) Return top-k2, mapping back to original indices
        """
        if not documents:
            return [], [], []

        # Step 1: TF-IDF filter
        tf_docs, tf_indices, _ = self.tfidf_rerank(query, documents, k=min(k1, len(documents)))

        # Step 2: Cross-encoder order within filtered subset
        ce_docs, ce_rel_indices, ce_scores = self.cross_encoder_rerank(query, tf_docs, k=min(k2, len(tf_docs)))

        # Map relative indices back to original document indices
        final_indices = [tf_indices[i] for i in ce_rel_indices]
        return ce_docs, final_indices, ce_scores
    
if __name__ == "__main__":
    query = "What are the health benefits of green tea?"
    documents = [
        "Green tea contains antioxidants that may help prevent cardiovascular disease.",
        "Coffee is also rich in antioxidants but can increase heart rate.",
        "Drinking water is essential for hydration.",
        "Green tea may also aid in weight loss and improve brain function."
    ]

    print("\nTF-IDF Reranking:")
    reranker = Reranker(type="tfidf")
    docs, indices, scores = reranker.rerank(query, documents)
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")

    print("\nCross-Encoder Reranking:")
    reranker = Reranker(type="cross_encoder")
    docs, indices, scores = reranker.rerank(query, documents)
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")

    print("\nHybrid Reranking:")
    reranker = Reranker(type="hybrid")
    docs, indices, scores = reranker.rerank(query, documents)
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")