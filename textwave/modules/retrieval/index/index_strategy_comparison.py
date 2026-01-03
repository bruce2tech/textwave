import pandas as pd
import numpy as np
import os
import re
import time
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from collections import defaultdict
import json

# Import your FAISS implementations
from bruteforce import FaissBruteForce
from lsh import FaissLSH
from hnsw import FaissHNSW

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
class IndexingResult:
    """Stores results for a single indexing strategy"""
    strategy_name: str
    build_time: float
    search_times: List[float]
    mean_search_time: float
    total_memory_mb: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]
    mean_reciprocal_rank: float
    num_questions: int

class SentenceChunking:
    """Sentence-based chunking - using the best strategy from Task 1"""
    
    def __init__(self, sentences_per_chunk: int = 5, min_chunk_size: int = 200):
        self.name = f"sentence_{sentences_per_chunk}_{min_chunk_size}"
        self.sentences_per_chunk = sentences_per_chunk
        self.min_chunk_size = min_chunk_size
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
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
            chunk_length = 0
            
            for j in range(self.sentences_per_chunk):
                if i + j < len(sentences):
                    sentence = sentences[i + j].strip()
                    if sentence:
                        chunk_sentences.append(sentence)
                        chunk_length += len(sentence)
            
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

class BaseIndexWrapper:
    """Base wrapper for FAISS-based indexing strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.index = None
        self.chunks = []
        self.build_time = 0.0
        self.vectorizer = None
        self.tfidf_matrix = None
    
    def build_index(self, chunks: List[Chunk], vectorizer: TfidfVectorizer):
        """Build the index"""
        raise NotImplementedError
    
    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Search for top-k most similar chunks"""
        raise NotImplementedError
    
    def get_memory_usage_mb(self) -> float:
        """Estimate memory usage in MB"""
        if self.tfidf_matrix is not None:
            matrix_memory = (self.tfidf_matrix.data.nbytes + 
                           self.tfidf_matrix.indices.nbytes + 
                           self.tfidf_matrix.indptr.nbytes) / (1024 * 1024)
            return matrix_memory + 50  # Add estimate for FAISS index overhead
        return 50  # Default estimate

class BruteForceWrapper(BaseIndexWrapper):
    """Wrapper for FAISS Brute Force implementation"""
    
    def __init__(self):
        super().__init__("FAISS Brute Force")
        
    def build_index(self, chunks: List[Chunk], vectorizer: TfidfVectorizer):
        """Build TF-IDF vectors and FAISS brute force index"""
        start_time = time.time()
        
        self.chunks = chunks
        self.vectorizer = vectorizer
        corpus = [chunk.text for chunk in chunks]
        self.tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Convert to dense for FAISS
        dense_vectors = self.tfidf_matrix.toarray().astype(np.float32)
        
        # Initialize FAISS brute force index with cosine similarity
        dim = dense_vectors.shape[1]
        self.index = FaissBruteForce(dim=dim, metric='cosine')
        
        # Create metadata (chunk indices)
        metadata = list(range(len(chunks)))
        
        # Add embeddings to FAISS index
        self.index.add_embeddings(dense_vectors.tolist(), metadata)
        
        self.build_time = time.time() - start_time
    
    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Search using FAISS brute force"""
        if self.index is None or self.vectorizer is None:
            return []
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        query_dense = query_vector.toarray().astype(np.float32)
        
        # Search with FAISS
        distances, indices = self.index.index.search(query_dense, k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.chunks):
                # Convert distance to similarity (for cosine: similarity = 1 - distance)
                similarity = 1.0 - dist if dist <= 1.0 else max(0.0, 2.0 - dist)
                results.append((idx, similarity))
        
        return results

class LSHWrapper(BaseIndexWrapper):
    """Wrapper for FAISS LSH implementation"""
    
    def __init__(self, nbits: int = 128):
        super().__init__(f"FAISS LSH (nbits={nbits})")
        self.nbits = nbits
        
    def build_index(self, chunks: List[Chunk], vectorizer: TfidfVectorizer):
        """Build TF-IDF vectors and FAISS LSH index"""
        start_time = time.time()
        
        self.chunks = chunks
        self.vectorizer = vectorizer
        corpus = [chunk.text for chunk in chunks]
        self.tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Convert to dense for FAISS
        dense_vectors = self.tfidf_matrix.toarray().astype(np.float32)
        
        # Initialize FAISS LSH index
        dim = dense_vectors.shape[1]
        self.index = FaissLSH(dim=dim, nbits=self.nbits, normalize=True)
        
        # Create metadata (chunk indices)
        metadata = list(range(len(chunks)))
        
        # Add embeddings to FAISS index
        self.index.add_embeddings(dense_vectors, metadata)
        
        self.build_time = time.time() - start_time
    
    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Search using FAISS LSH"""
        if self.index is None or self.vectorizer is None:
            return []
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        query_dense = query_vector.toarray().astype(np.float32)
        
        # Normalize query if index normalizes
        if self.index.normalize:
            import faiss
            faiss.normalize_L2(query_dense)
        
        # Search with FAISS LSH
        distances, indices = self.index.index.search(query_dense, k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.chunks):
                # For normalized vectors with inner product, distance is already similarity-like
                similarity = max(0.0, dist)  # LSH distances can be negative
                results.append((idx, similarity))
        
        return results

class HNSWWrapper(BaseIndexWrapper):
    """Wrapper for FAISS HNSW implementation"""
    
    def __init__(self, M: int = 16, efConstruction: int = 200, efSearch: int = 50):
        super().__init__(f"FAISS HNSW (M={M}, ef={efSearch})")
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        
    def build_index(self, chunks: List[Chunk], vectorizer: TfidfVectorizer):
        """Build TF-IDF vectors and FAISS HNSW index"""
        start_time = time.time()
        
        self.chunks = chunks
        self.vectorizer = vectorizer
        corpus = [chunk.text for chunk in chunks]
        self.tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Convert to dense for FAISS
        dense_vectors = self.tfidf_matrix.toarray().astype(np.float32)
        
        # Initialize FAISS HNSW index with cosine similarity
        dim = dense_vectors.shape[1]
        self.index = FaissHNSW(
            dim=dim, 
            M=self.M, 
            efConstruction=self.efConstruction, 
            efSearch=self.efSearch,
            metric='cosine',
            normalize=True
        )
        
        # Create metadata (chunk indices)
        metadata = list(range(len(chunks)))
        
        # Add embeddings to FAISS index
        self.index.add_embeddings(dense_vectors, metadata)
        
        self.build_time = time.time() - start_time
    
    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Search using FAISS HNSW"""
        if self.index is None or self.vectorizer is None:
            return []
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        query_dense = query_vector.toarray().astype(np.float32)
        
        # Normalize query if index normalizes
        if self.index.normalize:
            import faiss
            faiss.normalize_L2(query_dense)
        
        # Search with FAISS HNSW
        distances, indices = self.index.index.search(query_dense, k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.chunks):
                # Convert cosine distance to similarity
                similarity = 1.0 - dist if dist <= 1.0 else max(0.0, 2.0 - dist)
                results.append((idx, similarity))
        
        return results

class IndexingComparator:
    """Compare different indexing strategies using FAISS implementations"""
    
    def __init__(self, questions_file: str, documents_dir: str):
        self.questions_df = pd.read_csv(questions_file, sep='\t')
        self.documents_dir = documents_dir
        self.k_values = [1, 3, 5, 10]
        self.chunker = SentenceChunking(sentences_per_chunk=5, min_chunk_size=200)
        
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
    
    def prepare_data(self) -> Tuple[List[Chunk], TfidfVectorizer, List[Tuple[str, str]]]:
        """Prepare chunks and questions for evaluation"""
        print("Preparing data using sentence_5_200 chunking strategy...")
        
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
        
        # Prepare vectorizer
        vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            max_features=5000  # Reduced for FAISS efficiency
        )
        
        # Prepare questions
        questions = []
        for _, row in valid_questions.iterrows():
            if pd.notna(row['Question']) and pd.notna(row['ArticleFile']):
                questions.append((str(row['Question']), str(row['ArticleFile'])))
        
        print(f"Prepared {len(questions)} valid questions")
        
        return all_chunks, vectorizer, questions
    
    def evaluate_indexing_strategy(self, index_wrapper: BaseIndexWrapper, 
                                 chunks: List[Chunk], vectorizer: TfidfVectorizer,
                                 questions: List[Tuple[str, str]]) -> IndexingResult:
        """Evaluate a single indexing strategy"""
        print(f"\nEvaluating {index_wrapper.name}...")
        
        # Build index
        print(f"Building {index_wrapper.name} index...")
        index_wrapper.build_index(chunks, vectorizer)
        print(f"Index built in {index_wrapper.build_time:.2f} seconds")
        print(f"Estimated memory usage: {index_wrapper.get_memory_usage_mb():.2f} MB")
        
        # Evaluate questions
        search_times = []
        all_metrics = {k: {'precision': [], 'recall': [], 'hit_rate': []} for k in self.k_values}
        reciprocal_ranks = []
        
        print(f"Evaluating {len(questions)} questions...")
        
        for i, (question, target_doc) in enumerate(questions):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(questions)} questions")
            
            # Search
            start_time = time.time()
            results = index_wrapper.search(question, k=max(self.k_values))
            search_time = time.time() - start_time
            search_times.append(search_time)
            
            # Calculate metrics
            metrics = self._calculate_metrics(results, chunks, target_doc, self.k_values)
            
            for k in self.k_values:
                all_metrics[k]['precision'].append(metrics['precision_at_k'][k])
                all_metrics[k]['recall'].append(metrics['recall_at_k'][k])
                all_metrics[k]['hit_rate'].append(metrics['hit_rate_at_k'][k])
            
            reciprocal_ranks.append(metrics['reciprocal_rank'])
        
        # Aggregate results
        precision_at_k = {k: np.mean(all_metrics[k]['precision']) for k in self.k_values}
        recall_at_k = {k: np.mean(all_metrics[k]['recall']) for k in self.k_values}
        hit_rate_at_k = {k: np.mean(all_metrics[k]['hit_rate']) for k in self.k_values}
        
        return IndexingResult(
            strategy_name=index_wrapper.name,
            build_time=index_wrapper.build_time,
            search_times=search_times,
            mean_search_time=np.mean(search_times),
            total_memory_mb=index_wrapper.get_memory_usage_mb(),
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            hit_rate_at_k=hit_rate_at_k,
            mean_reciprocal_rank=np.mean(reciprocal_ranks),
            num_questions=len(questions)
        )
    
    def _calculate_metrics(self, results: List[Tuple[int, float]], chunks: List[Chunk], 
                          target_doc: str, k_values: List[int]) -> Dict[str, Any]:
        """Calculate evaluation metrics"""
        metrics = {
            'precision_at_k': {},
            'recall_at_k': {},
            'hit_rate_at_k': {},
            'reciprocal_rank': 0.0
        }
        
        # Find first relevant result
        first_relevant_pos = None
        for i, (chunk_idx, score) in enumerate(results):
            if chunk_idx < len(chunks) and chunks[chunk_idx].doc_id == target_doc:
                if first_relevant_pos is None:
                    first_relevant_pos = i + 1
                break
        
        if first_relevant_pos is not None:
            metrics['reciprocal_rank'] = 1.0 / first_relevant_pos
        
        # Calculate metrics for each k
        for k in k_values:
            top_k = results[:k]
            relevant_in_top_k = sum(1 for chunk_idx, score in top_k 
                                  if chunk_idx < len(chunks) and chunks[chunk_idx].doc_id == target_doc)
            
            metrics['precision_at_k'][k] = relevant_in_top_k / k if k > 0 else 0
            metrics['recall_at_k'][k] = min(relevant_in_top_k / 1.0, 1.0)  # Assuming 1 relevant doc
            metrics['hit_rate_at_k'][k] = relevant_in_top_k > 0
        
        return metrics
    
    def compare_strategies(self) -> Dict[str, IndexingResult]:
        """Compare all indexing strategies"""
        print("="*80)
        print("INDEXING STRATEGY COMPARISON")
        print("Using FAISS-based implementations")
        print("Chunking: sentence_5_200 (best from Task 1)")
        print("="*80)
        
        # Prepare data
        chunks, vectorizer, questions = self.prepare_data()
        
        # Initialize strategies
        strategies = [
            BruteForceWrapper(),
            LSHWrapper(nbits=128),
            HNSWWrapper(M=16, efConstruction=200, efSearch=50)
        ]
        
        # Evaluate each strategy
        results = {}
        for strategy in strategies:
            try:
                result = self.evaluate_indexing_strategy(strategy, chunks, vectorizer, questions)
                results[strategy.name] = result
            except Exception as e:
                print(f"Error evaluating {strategy.name}: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    
    def print_comparison(self, results: Dict[str, IndexingResult]):
        """Print detailed comparison results"""
        print("\n" + "="*100)
        print("INDEXING STRATEGY COMPARISON RESULTS")
        print("="*100)
        
        # Performance summary table
        print(f"\n{'Strategy':<20} {'Build Time':<12} {'Search Time':<14} {'Memory (MB)':<12} {'MRR':<8} {'P@1':<8} {'P@5':<8} {'HR@10':<8}")
        print("-" * 105)
        
        for strategy_name, result in results.items():
            print(f"{strategy_name:<20} {result.build_time:<12.3f} {result.mean_search_time*1000:<14.3f} "
                  f"{result.total_memory_mb:<12.1f} {result.mean_reciprocal_rank:<8.4f} "
                  f"{result.precision_at_k[1]:<8.4f} {result.precision_at_k[5]:<8.4f} "
                  f"{result.hit_rate_at_k[10]:<8.4f}")
        
        print("\nNotes:")
        print("- Build Time: Time to construct the index (seconds)")
        print("- Search Time: Average query time (milliseconds)")
        print("- Memory: Estimated memory usage (MB)")
        print("- MRR: Mean Reciprocal Rank")
        print("- P@K: Precision at K")
        print("- HR@K: Hit Rate at K")
        
        # Detailed results for each strategy
        for strategy_name, result in results.items():
            print(f"\n{strategy_name} - Detailed Results:")
            print("-" * 60)
            print(f"Build time: {result.build_time:.3f} seconds")
            print(f"Mean search time: {result.mean_search_time*1000:.3f} ms")
            print(f"Memory usage: {result.total_memory_mb:.1f} MB")
            print(f"Questions evaluated: {result.num_questions}")
            print(f"Mean Reciprocal Rank: {result.mean_reciprocal_rank:.4f}")
            
            print("\nPrecision@K:")
            for k in sorted(result.precision_at_k.keys()):
                print(f"  P@{k}: {result.precision_at_k[k]:.4f}")
            
            print("\nRecall@K:")
            for k in sorted(result.recall_at_k.keys()):
                print(f"  R@{k}: {result.recall_at_k[k]:.4f}")
            
            print("\nHit Rate@K:")
            for k in sorted(result.hit_rate_at_k.keys()):
                print(f"  HR@{k}: {result.hit_rate_at_k[k]:.4f}")

def main():
    """Main execution function"""
    # Configuration
    QUESTIONS_FILE = "textwave/qa_resources/question.tsv"
    DOCUMENTS_DIR = "textwave/storage/"
    
    # Initialize comparator
    comparator = IndexingComparator(QUESTIONS_FILE, DOCUMENTS_DIR)
    
    # Run comparison
    try:
        results = comparator.compare_strategies()
        comparator.print_comparison(results)
        
        # Save results
        results_dict = {}
        for name, result in results.items():
            results_dict[name] = {
                'strategy_name': result.strategy_name,
                'build_time': result.build_time,
                'mean_search_time': result.mean_search_time,
                'total_memory_mb': result.total_memory_mb,
                'precision_at_k': result.precision_at_k,
                'recall_at_k': result.recall_at_k,
                'hit_rate_at_k': result.hit_rate_at_k,
                'mean_reciprocal_rank': result.mean_reciprocal_rank,
                'num_questions': result.num_questions
            }
        
        with open('faiss_indexing_comparison_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nDetailed results saved to: faiss_indexing_comparison_results.json")
        
        # Print analysis and recommendations
        print("\n" + "="*80)
        print("ANALYSIS AND RECOMMENDATIONS")
        print("="*80)
        
        if results:
            # Find best performers
            best_accuracy = max(results.values(), key=lambda x: x.mean_reciprocal_rank)
            fastest_search = min(results.values(), key=lambda x: x.mean_search_time)
            lowest_memory = min(results.values(), key=lambda x: x.total_memory_mb)
            fastest_build = min(results.values(), key=lambda x: x.build_time)
            
            print(f"\nBest Accuracy (MRR): {best_accuracy.strategy_name} ({best_accuracy.mean_reciprocal_rank:.4f})")
            print(f"Fastest Search: {fastest_search.strategy_name} ({fastest_search.mean_search_time*1000:.3f} ms)")
            print(f"Lowest Memory: {lowest_memory.strategy_name} ({lowest_memory.total_memory_mb:.1f} MB)")
            print(f"Fastest Build: {fastest_build.strategy_name} ({fastest_build.build_time:.3f} seconds)")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()