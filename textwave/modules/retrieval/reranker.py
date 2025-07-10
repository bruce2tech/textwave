import os
import pickle

from sympy import vectorize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import torch
import numpy as np

# TODO: You will need to implement: 
#  - Reranker.cross_encoder_rerank()
#  - Reranker.tfidf_rerank()
#  - Reranker.bow_rerank()
#  - Reranker.sequential_rerank()


class Reranker:
    """
    Perform reranking of documents based on their relevance to a given query.

    Supports multiple reranking strategies:
    - Cross-encoder: Uses a transformer model to compute pairwise relevance.
    - TF-IDF: Uses term frequency-inverse document frequency with similarity metrics.
    - BoW: Uses term Bag-of-Words with similarity metrics.
    - Hybrid: Combines TF-IDF and cross-encoder scores.
    - Sequential: Applies TF-IDF first, then cross-encoder for refined reranking.
    """

    def __init__(self, type, cross_encoder_model_name='cross-encoder/ms-marco-TinyBERT-L-2-v2', corpus_directory=''):
        """
        Initialize the Reranker with a specified reranking strategy and optional model and corpus.

        :param type: Type of reranking ('cross_encoder', 'tfidf', 'bow', 'hybrid', or 'sequential').
        :param cross_encoder_model_name: HuggingFace model name for the cross-encoder (default: cross-encoder/ms-marco-TinyBERT-L-2-v2).
            - For more information on the default cross encoder, see https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2
            - For more information on general cross encoders, see https://huggingface.co/cross-encoder
        :param corpus_directory: Directory containing .txt files for TF-IDF corpus (optional).
        """
        self.type = type
        self.cross_encoder_model_name = cross_encoder_model_name
        self.cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model_name)


    def rerank(self, query, context, distance_metric="cosine", seq_k1=None, seq_k2=None):
        """
        Dispatch the reranking process based on the initialized strategy.

        :param query: Input query string to evaluate relevance against.
        :param context: List of document strings to rerank.
        :param distance_metric: Distance metric used for TF-IDF reranking (default: "cosine").
        :param seq_k1: Number of top documents to select in the first phase (TF-IDF) of sequential rerank.
        :param seq_k2: Number of top documents to return from the second phase (cross-encoder) of sequential rerank.
        :return: Tuple of (ranked documents, ranked indices, corresponding scores).
        """
        if self.type == "cross_encoder":
            return self.cross_encoder_rerank(query, context)
        elif self.type == "tfidf":
            return self.tfidf_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "bow":
            return self.bow_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "hybrid":
            return self.hybrid_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "sequential":
            return self.sequential_rerank(query, context, seq_k1, seq_k2, distance_metric=distance_metric)

    def cross_encoder_rerank(self, query, context):
        """
        Rerank documents using a cross-encoder transformer model.

        Computes relevance scores for each document-query pair, sorts them in
        descending order of relevance, and returns the ranked results.

        NOTE: See https://huggingface.co/cross-encoder for more information on 
        implementing cross-encoder

        :param query: Query string.
        :param context: List of candidate document strings.
        :return: Tuple of (ranked documents, ranked indices, relevance scores).
        """
        query_document_pairs = [(query, doc) for doc in context]
        inputs = self.tokenizer(query_document_pairs, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            logits = self.cross_encoder_model(**inputs).logits
            relevance_scores = logits.squeeze().tolist()

        #########################################
        # TODO: Implement cross_encoder_rerank()
        #########################################
        
        pass



    def tfidf_rerank(self, query, context, distance_metric="cosine"):
        """
        Rerank documents using TF-IDF vectorization and distance-based similarity.

        Creates a TF-IDF matrix from the query and context, computes pairwise distances,
        and sorts documents by similarity (lower distance implies higher relevance).

        :param query: Query string.
        :param context: List of document strings.
        :param distance_metric: Distance function to use (e.g., 'cosine', 'euclidean').
        :return: Tuple of (ranked documents, indices, similarity scores).
        """
        #########################################
        # TODO: Implement tfidf_rerank()
        #########################################
        pass

    def bow_rerank(self, query, context, distance_metric="cosine"):
        """
        Rerank documents using BoW vectorization and distance-based similarity.

        Creates a BoW matrix from the query and context, computes pairwise distances,
        and sorts documents by similarity (lower distance implies higher relevance).

        :param query: Query string.
        :param context: List of document strings.
        :param distance_metric: Distance function to use (e.g., 'cosine', 'euclidean').
        :return: Tuple of (ranked documents, indices, similarity scores).
        """
        #########################################
        # TODO: Implement bow_rerank()
        #########################################
        pass


    def hybrid_rerank(self, query, context, distance_metric="cosine", tfidf_weight=0.3):
        """
        Combine TF-IDF and cross-encoder scores to produce a hybrid reranking.

        This approach balances fast lexical matching (TF-IDF) with deeper semantic understanding
        (cross-encoder) by computing a weighted average of both scores.

        :param query: Query string.
        :param context: List of document strings.
        :param distance_metric: Distance metric for the TF-IDF portion.
        :param tfidf_weight: Weight (0-1) assigned to TF-IDF score in final ranking.
        :return: Tuple of (ranked documents, indices, combined scores).
        """
        #########################################
        # TODO: Implement hybrid_rerank()
        #########################################
        pass

    def sequential_rerank(self, query, context, seq_k1, seq_k2, distance_metric="cosine"):
        """
        Apply a two-stage reranking pipeline: TF-IDF followed by cross-encoder.

        This method narrows down the document pool using TF-IDF, then applies a
        cross-encoder to refine the top-k results for improved relevance accuracy.

        :param query: Query string.
        :param context: List of document strings.
        :param seq_k1: Top-k documents to retain after the first stage (TF-IDF).
        :param seq_k2: Final top-k documents to return after second stage (cross-encoder).
        :param distance_metric: Distance metric for TF-IDF.
        :return: Tuple of (ranked documents, indices, final relevance scores).
        """
        #########################################
        # TODO: Implement sequential_rerank()
        #########################################
        pass


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
