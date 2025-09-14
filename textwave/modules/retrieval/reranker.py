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
        if not context:
            return [], [], []
        
        query_document_pairs = [(query, doc) for doc in context]
        inputs = self.tokenizer(query_document_pairs, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            logits = self.cross_encoder_model(**inputs).logits
            
            # Handle tensor shape properly
            if logits.dim() > 1:
                relevance_scores = logits.squeeze(-1)
            else:
                relevance_scores = logits
                
            # Convert to list and ensure we have correct number of scores
            if relevance_scores.dim() == 0:
                score_list = [float(relevance_scores)]
            else:
                score_list = relevance_scores.tolist()
            
            # Take only the scores for our context and round for precision
            relevance_scores = [round(float(score), 1) for score in score_list[:len(context)]]
        
        # Ensure we have the right number of scores
        if len(relevance_scores) != len(context):
            # Fallback: pad with zeros or truncate
            if len(relevance_scores) < len(context):
                relevance_scores.extend([0.0] * (len(context) - len(relevance_scores)))
            else:
                relevance_scores = relevance_scores[:len(context)]
        
        # Create (score, index) pairs and sort by score in descending order
        scored_docs = [(relevance_scores[i], i) for i in range(len(context))]
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Extract ranked results
        ranked_indices = [idx for _, idx in scored_docs]
        ranked_scores = [score for score, _ in scored_docs]
        ranked_documents = [context[idx] for idx in ranked_indices]
        
        return ranked_documents, ranked_indices, ranked_scores

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
        if not context:
            return [], [], []
        
        # Combine query and documents for TF-IDF fitting
        all_docs = [query] + context
        
        # Create TF-IDF vectorizer and fit on all documents
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        tfidf_matrix = vectorizer.fit_transform(all_docs)
        
        # Extract query vector (first row) and document vectors (remaining rows)
        query_vector = tfidf_matrix[0:1]
        doc_vectors = tfidf_matrix[1:]
        
        # Compute pairwise distances between query and documents
        distances = pairwise_distances(query_vector, doc_vectors, metric=distance_metric)
        distances = distances.flatten()
        
        # Create (distance, index, doc) tuples and sort by distance (ascending = better)
        scored_items = [(dist, idx, context[idx]) for idx, dist in enumerate(distances)]
        scored_items.sort(key=lambda x: x[0])  # Sort by distance (ascending)
        
        # Extract results
        ranked_distances = [item[0] for item in scored_items]
        ranked_indices = [item[1] for item in scored_items]
        ranked_documents = [item[2] for item in scored_items]
        
        return ranked_documents, ranked_indices, ranked_distances


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
        if not context:
            return [], [], []
        
        # Import the BagOfWords class
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
        from modules.utils.bow import BagOfWords
        
        # Create and fit BoW model
        bow = BagOfWords()
        bow.fit([query] + context)
        
        # Transform query and documents to BoW vectors
        query_vector = bow.transform(query).reshape(1, -1)
        doc_vectors = np.array([bow.transform(doc) for doc in context])
        
        # Compute pairwise distances between query and documents
        distances = pairwise_distances(query_vector, doc_vectors, metric=distance_metric)
        distances = distances.flatten()
        
        # Create (distance, index, doc) tuples and sort by distance (ascending = better)
        scored_items = [(dist, idx, context[idx]) for idx, dist in enumerate(distances)]
        scored_items.sort(key=lambda x: x[0])  # Sort by distance (ascending)
        
        # Extract results
        ranked_distances = [item[0] for item in scored_items]
        ranked_indices = [item[1] for item in scored_items]
        ranked_documents = [item[2] for item in scored_items]
        
        return ranked_documents, ranked_indices, ranked_distances


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
        if not context:
            return [], [], []
        
        print(f"DEBUG Sequential: Starting with {len(context)} documents")
        print(f"DEBUG Sequential: Query: '{query}'")
        print(f"DEBUG Sequential: Documents: {context}")
        
        # Stage 1: TF-IDF reranking to get top seq_k1 documents
        stage1_docs, stage1_indices, stage1_scores = self.tfidf_rerank(query, context, distance_metric)
        
        print(f"DEBUG Sequential: TF-IDF ranked docs: {stage1_docs}")
        print(f"DEBUG Sequential: TF-IDF indices: {stage1_indices}")
        print(f"DEBUG Sequential: TF-IDF scores: {stage1_scores}")
        
        # Take top seq_k1 documents from stage 1
        k1 = min(seq_k1 or len(stage1_docs), len(stage1_docs))
        top_k1_docs = stage1_docs[:k1]
        top_k1_indices = stage1_indices[:k1]
        
        print(f"DEBUG Sequential: Top {k1} docs for stage 2: {top_k1_docs}")
        print(f"DEBUG Sequential: Top {k1} indices: {top_k1_indices}")
        
        # Stage 2: Cross-encoder reranking on the top seq_k1 documents
        if top_k1_docs:
            stage2_docs, stage2_relative_indices, stage2_scores = self.cross_encoder_rerank(query, top_k1_docs)
            
            print(f"DEBUG Sequential: Cross-encoder ranked docs: {stage2_docs}")
            print(f"DEBUG Sequential: Cross-encoder relative indices: {stage2_relative_indices}")
            print(f"DEBUG Sequential: Cross-encoder scores: {stage2_scores}")
            
            # Map the relative indices from stage 2 back to original context indices
            stage2_original_indices = [top_k1_indices[rel_idx] for rel_idx in stage2_relative_indices]
            
            # Take final top seq_k2 documents
            k2 = min(seq_k2 or len(stage2_docs), len(stage2_docs))
            final_docs = stage2_docs[:k2]
            final_indices = stage2_original_indices[:k2]
            final_scores = stage2_scores[:k2]
            
            print(f"DEBUG Sequential: Final top {k2} docs: {final_docs}")
            print(f"DEBUG Sequential: Final indices: {final_indices}")
            print(f"DEBUG Sequential: Final scores: {final_scores}")
            
            return final_docs, final_indices, final_scores
        else:
            return [], [], []

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
