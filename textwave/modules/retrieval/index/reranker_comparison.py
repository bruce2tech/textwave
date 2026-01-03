import pandas as pd
import numpy as np
import os
import sys
import time
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import nltk
import json
import math

# Simple imports - restore FAISS HNSW
sys.path.append('..')
sys.path.append('.')

try:
    from reranker import Reranker
    from hnsw import FaissHNSW
    MODULES_AVAILABLE = True
    print("Successfully imported reranker and hnsw")
except ImportError as e:
    print(f"Could not import modules: {e}")
    MODULES_AVAILABLE = False

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    doc_id: str
    chunk_id: int
    start_pos: int
    end_pos: int
    strategy: str

@dataclass
class SearchResult:
    """Represents a search result with original and re-ranked scores"""
    chunk_idx: int
    chunk: Chunk
    original_score: float
    reranked_score: Optional[float] = None
    final_rank: Optional[int] = None

@dataclass
class ReRankingResult:
    """Stores results for a single re-ranking strategy"""
    strategy_name: str
    rerank_time: float
    mean_rerank_time: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]
    mean_reciprocal_rank: float
    ndcg_at_k: Dict[int, float]
    num_questions: int

class SentenceChunking:
    """Sentence-based chunking"""
    
    def __init__(self, sentences_per_chunk: int = 5, min_chunk_size: int = 200):
        self.name = f"sentence_{sentences_per_chunk}_{min_chunk_size}"
        self.sentences_per_chunk = sentences_per_chunk
        self.min_chunk_size = min_chunk_size
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', ' ', text)
        return text.strip()
    
    def chunk_document(self, text: str, doc_id: str) -> List[Chunk]:
        text = self.preprocess_text(text)
        sentences = nltk.sent_tokenize(text)
        chunks = []
        chunk_id = 0
        
        i = 0
        while i < len(sentences):
            chunk_sentences = []
            for j in range(self.sentences_per_chunk):
                if i + j < len(sentences):
                    sentence = sentences[i + j].strip()
                    if sentence:
                        chunk_sentences.append(sentence)
            
            if chunk_sentences:
                chunk_text = ' '.join(chunk_sentences)
                if len(chunk_text) >= self.min_chunk_size:
                    start_pos = text.find(chunk_sentences[0])
                    end_pos = start_pos + len(chunk_text)
                    
                    chunks.append(Chunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        start_pos=start_pos if start_pos >= 0 else 0,
                        end_pos=end_pos,
                        strategy=self.name
                    ))
                    chunk_id += 1
            
            i += self.sentences_per_chunk
        
        return chunks

class HNSWRetriever:
    """FAISS HNSW based retrieval system with OpenMP disabled"""
    
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Disable OpenMP threading to avoid conflicts
        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            max_features=2000  # Reduced for stability
        )
        
        # Build TF-IDF matrix
        corpus = [chunk.text for chunk in chunks]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        dense_vectors = tfidf_matrix.toarray().astype(np.float32)
        
        # Initialize FAISS HNSW index with conservative settings
        dim = dense_vectors.shape[1]
        
        try:
            self.index = FaissHNSW(
                dim=dim, 
                M=8,  # Reduced from 16
                efConstruction=40,  # Reduced from 200
                efSearch=16,  # Reduced from 50
                metric='cosine',
                normalize=True
            )
            
            # Add embeddings to index
            metadata = list(range(len(chunks)))
            self.index.add_embeddings(dense_vectors, metadata)
            print(f"Built FAISS HNSW index with {len(chunks)} chunks (OpenMP disabled)")
            
        except Exception as e:
            print(f"Error building FAISS HNSW index: {e}")
            print("Falling back to simple similarity search")
            self.index = None
            self.tfidf_matrix = tfidf_matrix
    
    def search(self, query: str, k: int = 20) -> List[SearchResult]:
        """Search for top-k most similar chunks"""
        if self.index is not None:
            return self._search_hnsw(query, k)
        else:
            return self._search_simple(query, k)
    
    def _search_hnsw(self, query: str, k: int) -> List[SearchResult]:
        """Search using FAISS HNSW"""
        # Transform query
        query_vector = self.vectorizer.transform([query])
        query_dense = query_vector.toarray().astype(np.float32)
        
        # Normalize query if needed
        if self.index.normalize:
            import faiss
            faiss.normalize_L2(query_dense)
        
        # Search with FAISS HNSW
        distances, indices = self.index.index.search(query_dense, k)
        
        # Create SearchResult objects
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.chunks):
                # Convert cosine distance to similarity
                similarity = 1.0 - dist if dist <= 1.0 else max(0.0, 2.0 - dist)
                result = SearchResult(
                    chunk_idx=idx,
                    chunk=self.chunks[idx],
                    original_score=similarity
                )
                results.append(result)
        
        return results
    
    def _search_simple(self, query: str, k: int) -> List[SearchResult]:
        """Fallback simple cosine similarity search"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Create SearchResult objects
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                result = SearchResult(
                    chunk_idx=idx,
                    chunk=self.chunks[idx],
                    original_score=similarities[idx]
                )
                results.append(result)
        
        return results

class BaseReRanker:
    """Base wrapper for re-ranking methods"""
    
    def __init__(self, name: str):
        self.name = name
    
    def rerank(self, query: str, results: List[SearchResult], k: int = 10) -> List[SearchResult]:
        """Re-rank the search results"""
        return results[:k]

class NoReRanking(BaseReRanker):
    """Baseline: No re-ranking"""
    
    def __init__(self):
        super().__init__("No Re-ranking (Baseline)")
    
    def rerank(self, query: str, results: List[SearchResult], k: int = 10) -> List[SearchResult]:
        """Return results in original order"""
        limited_results = results[:k]
        for i, result in enumerate(limited_results):
            result.reranked_score = result.original_score
            result.final_rank = i + 1
        return limited_results

class RerankerWrapper(BaseReRanker):
    """Wrapper for your Reranker implementations"""
    
    def __init__(self, reranker_type: str, **kwargs):
        self.reranker_type = reranker_type
        self.kwargs = kwargs
        
        type_names = {
            "tfidf": "TF-IDF Re-ranking",
            "bow": "Bag-of-Words Re-ranking",
            "cross_encoder": "Cross-Encoder Re-ranking",
            "hybrid": "Hybrid Re-ranking",
            "sequential": "Sequential Re-ranking"
        }
        
        super().__init__(type_names.get(reranker_type, reranker_type))
        
        # Initialize the reranker
        if MODULES_AVAILABLE:
            try:
                self.reranker = Reranker(type=reranker_type, **kwargs)
                print(f"Initialized {self.name}")
            except Exception as e:
                print(f"Warning: Could not initialize {reranker_type}: {e}")
                self.reranker = None
        else:
            self.reranker = None
    
    def rerank(self, query: str, results: List[SearchResult], k: int = 10) -> List[SearchResult]:
        """Re-rank using your Reranker class"""
        if not results or self.reranker is None:
            return results[:k]
        
        try:
            # Extract context documents
            context = [result.chunk.text for result in results]
            
            # Apply re-ranking
            if self.reranker_type == "sequential":
                # Get seq_k values from the wrapper or use defaults
                seq_k1 = getattr(self, 'seq_k1', min(10, len(results)))
                seq_k2 = getattr(self, 'seq_k2', k)
                
                reranked_docs, reranked_indices, reranked_scores = self.reranker.rerank(
                    query, context, seq_k1=seq_k1, seq_k2=seq_k2
                )
                
            else:
                reranked_docs, reranked_indices, reranked_scores = self.reranker.rerank(query, context)
            
            # Create new results list
            reranked_results = []
            for i, (doc, orig_idx, score) in enumerate(zip(reranked_docs, reranked_indices, reranked_scores)):
                if i >= k:
                    break
                
                if orig_idx >= len(results):
                    continue
                
                result = results[orig_idx]
                result.reranked_score = float(score)
                result.final_rank = i + 1
                reranked_results.append(result)
            
            return reranked_results
            
        except Exception as e:
            print(f"Warning: Re-ranking failed for {self.name}: {e}")
            # Return original results if re-ranking fails
            limited_results = results[:k]
            for i, result in enumerate(limited_results):
                result.reranked_score = result.original_score
                result.final_rank = i + 1
            return limited_results

class ReRankingComparator:
    """Compare different re-ranking strategies using simple retrieval"""
    
    def __init__(self):
        # Simple file paths from the script location
        self.questions_file = "../../../qa_resources/question.tsv"
        self.documents_dir = "../../../storage/"
        
        self.k_values = [1, 3, 5, 10]
        self.chunker = SentenceChunking(sentences_per_chunk=5, min_chunk_size=200)
        
        # Initialize base retriever (avoiding FAISS)
        self.retriever = None
        self.chunks = []
        
    def load_document(self, filename: str) -> str:
        """Load document with robust encoding handling"""
        if pd.isna(filename) or filename == 'nan':
            return ""
        
        if not filename.endswith('.txt.clean'):
            filename = filename + '.txt.clean'
        
        filepath = os.path.join(self.documents_dir, filename)
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    return f.read()
            except (FileNotFoundError, UnicodeDecodeError):
                continue
        
        return ""
    
    def prepare_base_system(self) -> Tuple[List[Chunk], List[Tuple[str, str]]]:
        """Prepare the base retrieval system using FAISS HNSW"""
        if not MODULES_AVAILABLE:
            raise ImportError("Required modules not available")
            
        print("Preparing base retrieval system...")
        print("Using: FAISS HNSW + sentence_5_200 chunking")
        
        # Load questions
        try:
            self.questions_df = pd.read_csv(self.questions_file, sep='\t')
        except FileNotFoundError:
            print(f"Error: Could not find questions file at {self.questions_file}")
            print("Current working directory:", os.getcwd())
            raise
        
        # Create chunks
        all_chunks = []
        valid_questions = self.questions_df.dropna(subset=['ArticleFile', 'Question'])
        unique_docs = valid_questions['ArticleFile'].unique()
        
        for doc_file in unique_docs:
            doc_text = self.load_document(doc_file)
            if doc_text:
                chunks = self.chunker.chunk_document(doc_text, doc_file)
                all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks from {len(unique_docs)} documents")
        
        # Initialize HNSW retriever
        self.retriever = HNSWRetriever(all_chunks)
        self.chunks = all_chunks
        
        # Prepare questions
        questions = []
        for _, row in valid_questions.iterrows():
            if pd.notna(row['Question']) and pd.notna(row['ArticleFile']):
                questions.append((str(row['Question']), str(row['ArticleFile'])))
        
        print(f"Prepared {len(questions)} valid questions")
        print("FAISS HNSW system built successfully\n")
        
        return all_chunks, questions
    
    def evaluate_reranker(self, reranker, questions: List[Tuple[str, str]]) -> ReRankingResult:
        """Evaluate a single re-ranking strategy"""
        print(f"Evaluating {reranker.name}...")
        
        rerank_times = []
        all_metrics = {k: {'precision': [], 'recall': [], 'hit_rate': [], 'ndcg': []} for k in self.k_values}
        reciprocal_ranks = []
        
        for i, (question, target_doc) in enumerate(questions):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(questions)} questions")
            
            # Get initial results
            initial_results = self.retriever.search(question, k=20)
            
            if not initial_results:
                continue
            
            # Apply re-ranking
            start_time = time.time()
            reranked_results = reranker.rerank(question, initial_results.copy(), k=max(self.k_values))
            rerank_time = time.time() - start_time
            rerank_times.append(rerank_time)
            
            # Calculate metrics
            metrics = self._calculate_metrics(reranked_results, target_doc, self.k_values)
            
            for k in self.k_values:
                all_metrics[k]['precision'].append(metrics['precision_at_k'][k])
                all_metrics[k]['recall'].append(metrics['recall_at_k'][k])
                all_metrics[k]['hit_rate'].append(metrics['hit_rate_at_k'][k])
                all_metrics[k]['ndcg'].append(metrics['ndcg_at_k'][k])
            
            reciprocal_ranks.append(metrics['reciprocal_rank'])
        
        # Aggregate results
        precision_at_k = {k: np.mean(all_metrics[k]['precision']) for k in self.k_values}
        recall_at_k = {k: np.mean(all_metrics[k]['recall']) for k in self.k_values}
        hit_rate_at_k = {k: np.mean(all_metrics[k]['hit_rate']) for k in self.k_values}
        ndcg_at_k = {k: np.mean(all_metrics[k]['ndcg']) for k in self.k_values}
        
        return ReRankingResult(
            strategy_name=reranker.name,
            rerank_time=sum(rerank_times),
            mean_rerank_time=np.mean(rerank_times) if rerank_times else 0.0,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            hit_rate_at_k=hit_rate_at_k,
            mean_reciprocal_rank=np.mean(reciprocal_ranks),
            ndcg_at_k=ndcg_at_k,
            num_questions=len(questions)
        )
    
    def _calculate_metrics(self, results: List[SearchResult], target_doc: str, 
                          k_values: List[int]) -> Dict[str, Any]:
        """Calculate evaluation metrics"""
        metrics = {
            'precision_at_k': {},
            'recall_at_k': {},
            'hit_rate_at_k': {},
            'ndcg_at_k': {},
            'reciprocal_rank': 0.0
        }
        
        # Create relevance list
        relevance = [1 if result.chunk.doc_id == target_doc else 0 for result in results]
        
        # Find first relevant result
        first_relevant_pos = None
        for i, rel in enumerate(relevance):
            if rel == 1:
                first_relevant_pos = i + 1
                break
        
        if first_relevant_pos is not None:
            metrics['reciprocal_rank'] = 1.0 / first_relevant_pos
        
        # Calculate metrics for each k
        for k in k_values:
            k_limited = min(k, len(results))  # Don't go beyond available results
            
            relevant_at_k = sum(relevance[:k_limited])
            metrics['precision_at_k'][k] = relevant_at_k / k_limited if k_limited > 0 else 0
            metrics['recall_at_k'][k] = min(relevant_at_k / 1.0, 1.0)
            metrics['hit_rate_at_k'][k] = relevant_at_k > 0
            metrics['ndcg_at_k'][k] = self._calculate_ndcg(relevance[:k_limited])
        
        return metrics
    
    def _calculate_ndcg(self, relevance: List[int]) -> float:
        """Calculate NDCG"""
        if not relevance:
            return 0.0
        
        # DCG
        dcg = relevance[0]
        for i in range(1, len(relevance)):
            dcg += relevance[i] / math.log2(i + 1)
        
        # IDCG
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = ideal_relevance[0] if ideal_relevance else 0
        for i in range(1, len(ideal_relevance)):
            idcg += ideal_relevance[i] / math.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def run_comparison(self) -> Dict[str, ReRankingResult]:
        """Run the full comparison"""
        print("="*80)
        print("RE-RANKING METHODS COMPARISON")
        print("Base System: FAISS HNSW + sentence_5_200 chunking")
        print("="*80)
        
        if not MODULES_AVAILABLE:
            print("ERROR: Required modules not available")
            return {}
        
        try:
            # Prepare base system
            chunks, questions = self.prepare_base_system()
            
            # Initialize re-ranking strategies (removed hybrid)
            rerankers = [
                NoReRanking(),
                RerankerWrapper("tfidf"),
                RerankerWrapper("bow"),
            ]
            
            # Add advanced re-rankers if they initialize successfully
            advanced_rerankers = [
                ("cross_encoder", {}),
                ("sequential", {})  # Remove seq_k1 and seq_k2 from init
            ]
            
            for reranker_type, kwargs in advanced_rerankers:
                try:
                    if reranker_type == "sequential":
                        # For sequential, we'll store the k values separately
                        reranker = RerankerWrapper(reranker_type, **kwargs)
                        if hasattr(reranker, 'reranker') and reranker.reranker is not None:
                            # Store seq_k values for use in rerank method
                            reranker.seq_k1 = 10
                            reranker.seq_k2 = 5
                            rerankers.append(reranker)
                    else:
                        reranker = RerankerWrapper(reranker_type, **kwargs)
                        if hasattr(reranker, 'reranker') and reranker.reranker is not None:
                            rerankers.append(reranker)
                except Exception as e:
                    print(f"Skipping {reranker_type}: {e}")
            
            # Evaluate each re-ranker
            results = {}
            for reranker in rerankers:
                try:
                    result = self.evaluate_reranker(reranker, questions)
                    results[reranker.name] = result
                except Exception as e:
                    print(f"Error evaluating {reranker.name}: {e}")
            
            return results
            
        except Exception as e:
            print(f"Error during comparison: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def print_results(self, results: Dict[str, ReRankingResult]):
        """Print comparison results"""
        if not results:
            print("No results to display")
            return
            
        print("\n" + "="*100)
        print("RE-RANKING COMPARISON RESULTS")
        print("="*100)
        
        # Summary table
        print(f"\n{'Strategy':<30} {'Rerank Time':<12} {'MRR':<8} {'P@1':<8} {'P@5':<8} {'HR@10':<8}")
        print("-" * 90)
        
        for strategy_name, result in results.items():
            print(f"{strategy_name:<30} {result.mean_rerank_time*1000:<12.2f} "
                  f"{result.mean_reciprocal_rank:<8.4f} {result.precision_at_k[1]:<8.4f} "
                  f"{result.precision_at_k[5]:<8.4f} {result.hit_rate_at_k[10]:<8.4f}")
        
        # Find best performers
        if results:
            best_accuracy = max(results.values(), key=lambda x: x.mean_reciprocal_rank)
            fastest_rerank = min(results.values(), key=lambda x: x.mean_rerank_time)
            
            print(f"\nBest Overall (MRR): {best_accuracy.strategy_name} ({best_accuracy.mean_reciprocal_rank:.4f})")
            print(f"Fastest Re-ranking: {fastest_rerank.strategy_name} ({fastest_rerank.mean_rerank_time*1000:.2f} ms)")
        
        # Save results
        results_dict = {}
        for name, result in results.items():
            results_dict[name] = {
                'strategy_name': result.strategy_name,
                'mean_rerank_time': result.mean_rerank_time,
                'precision_at_k': result.precision_at_k,
                'recall_at_k': result.recall_at_k,
                'hit_rate_at_k': result.hit_rate_at_k,
                'ndcg_at_k': result.ndcg_at_k,
                'mean_reciprocal_rank': result.mean_reciprocal_rank,
                'num_questions': result.num_questions
            }
        
        with open('reranking_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: reranking_results.json")

def main():
    """Main execution function"""
    print("Current working directory:", os.getcwd())
    print("Expected to be run from: textwave/modules/retrieval/index/")
    print("Using FAISS HNSW for base retrieval system")
    print()
    
    comparator = ReRankingComparator()
    results = comparator.run_comparison()
    comparator.print_results(results)

if __name__ == "__main__":
    main()