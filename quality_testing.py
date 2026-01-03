import pandas as pd
import numpy as np
import os
import re
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from collections import defaultdict
import json

# Download required NLTK data (run once)
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
class EvaluationResult:
    """Stores evaluation metrics for a single question"""
    question_id: int
    question_text: str
    target_doc: str
    retrieved_chunks: List[Tuple[Chunk, float]]  # (chunk, similarity_score)
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    reciprocal_rank: float
    hit_rate_at_k: Dict[int, bool]

class ChunkingStrategy:
    """Base class for chunking strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def chunk_document(self, text: str, doc_id: str) -> List[Chunk]:
        """Split document into chunks"""
        raise NotImplementedError
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation for sentences
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', ' ', text)
        return text.strip()

class FixedLengthChunking(ChunkingStrategy):
    """Fixed-length chunking with overlap"""
    
    def __init__(self, chunk_size: int = 1024, overlap: int = 128):
        super().__init__(f"fixed_{chunk_size}_{overlap}")
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, text: str, doc_id: str) -> List[Chunk]:
        text = self.preprocess_text(text)
        chunks = []
        
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at word boundary if not at end of text
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 0:
                chunks.append(Chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    start_pos=start,
                    end_pos=end,
                    strategy=self.name
                ))
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.overlap if end - self.overlap > start else end
            
            if start >= len(text):
                break
        
        return chunks

class SentenceChunking(ChunkingStrategy):
    """Sentence-based chunking"""
    
    def __init__(self, sentences_per_chunk: int = 4, min_chunk_size: int = 100):
        super().__init__(f"sentence_{sentences_per_chunk}_{min_chunk_size}")
        self.sentences_per_chunk = sentences_per_chunk
        self.min_chunk_size = min_chunk_size
    
    def chunk_document(self, text: str, doc_id: str) -> List[Chunk]:
        text = self.preprocess_text(text)
        
        # Split into sentences using NLTK
        sentences = nltk.sent_tokenize(text)
        chunks = []
        chunk_id = 0
        
        i = 0
        while i < len(sentences):
            # Collect sentences for this chunk
            chunk_sentences = []
            chunk_length = 0
            
            # Add sentences until we reach the target count or minimum size
            for j in range(self.sentences_per_chunk):
                if i + j < len(sentences):
                    sentence = sentences[i + j].strip()
                    if sentence:
                        chunk_sentences.append(sentence)
                        chunk_length += len(sentence)
            
            if chunk_sentences:
                chunk_text = ' '.join(chunk_sentences)
                
                # Only create chunk if it meets minimum size requirement
                if len(chunk_text) >= self.min_chunk_size:
                    # Find positions in original text
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

class BruteForceRetriever:
    """Simple TF-IDF based retrieval"""
    
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),  # Include bigrams
            max_features=10000
        )
        
        # Build corpus
        corpus = [chunk.text for chunk in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Chunk, float]]:
        """Search for top-k most similar chunks"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include chunks with positive similarity
                results.append((self.chunks[idx], similarities[idx]))
        
        return results

class ChunkingEvaluator:
    """Main evaluation class"""
    
    def __init__(self, questions_file: str, documents_dir: str):
        self.questions_df = pd.read_csv(questions_file, sep='\t')
        self.documents_dir = documents_dir
        self.k_values = [1, 3, 5, 10]
        
    def load_document(self, filename: str) -> str:
        """Load document text with robust encoding handling"""
        # Handle NaN values
        if pd.isna(filename) or filename == 'nan':
            print(f"Warning: Invalid filename (NaN)")
            return ""
        
        # Add .txt.clean extension if not present
        if not filename.endswith('.txt.clean'):
            filename = filename + '.txt.clean'
        
        filepath = os.path.join(self.documents_dir, filename)
        
        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                    # If we used a non-UTF-8 encoding, try to normalize to UTF-8
                    if encoding != 'utf-8':
                        print(f"Note: {filename} loaded with {encoding} encoding")
                    return content
            except FileNotFoundError:
                print(f"Warning: Document {filename} not found at {filepath}")
                return ""
            except UnicodeDecodeError:
                continue  # Try next encoding
        
        # If all encodings fail, try reading as binary and replacing errors
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                print(f"Warning: {filename} had encoding issues, some characters may be corrupted")
                return content
        except FileNotFoundError:
            print(f"Warning: Document {filename} not found at {filepath}")
            return ""
        except Exception as e:
            print(f"Warning: Could not read {filename}: {e}")
            return ""
    
    def evaluate_strategy(self, strategy: ChunkingStrategy) -> List[EvaluationResult]:
        """Evaluate a single chunking strategy"""
        print(f"\nEvaluating strategy: {strategy.name}")
        
        # Create chunks for all documents
        all_chunks = []
        document_chunks = {}  # doc_id -> list of chunks
        
        # Filter out NaN values from ArticleFile column
        valid_questions = self.questions_df.dropna(subset=['ArticleFile'])
        unique_docs = valid_questions['ArticleFile'].unique()
        
        for doc_file in unique_docs:
            print(f"Processing document: {doc_file}")
            doc_text = self.load_document(doc_file)
            if doc_text:
                chunks = strategy.chunk_document(doc_text, doc_file)
                all_chunks.extend(chunks)
                document_chunks[doc_file] = chunks
                print(f"  Created {len(chunks)} chunks")
            else:
                print(f"  No content loaded for {doc_file}")
        
        print(f"Created {len(all_chunks)} chunks total")
        
        if not all_chunks:
            print("No chunks created - check document paths")
            return []
        
        # Initialize retriever
        retriever = BruteForceRetriever(all_chunks)
        
        # Evaluate each question (only those with valid ArticleFile)
        results = []
        for idx, row in valid_questions.iterrows():
            question = row['Question']
            target_doc = row['ArticleFile']
            
            # Double-check that target_doc is valid (extra safety)
            if pd.isna(target_doc) or target_doc == 'nan' or not str(target_doc).strip():
                print(f"Skipping question {idx} due to invalid target document")
                continue
            
            # Also check that question is valid
            if pd.isna(question) or not str(question).strip():
                print(f"Skipping question {idx} due to invalid question text")
                continue
            
            # Retrieve chunks
            retrieved = retriever.search(str(question), k=max(self.k_values))
            
            # Calculate metrics
            metrics = self._calculate_metrics(retrieved, str(target_doc), self.k_values)
            
            result = EvaluationResult(
                question_id=idx,
                question_text=str(question),
                target_doc=str(target_doc),
                retrieved_chunks=retrieved,
                precision_at_k=metrics['precision_at_k'],
                recall_at_k=metrics['recall_at_k'],
                reciprocal_rank=metrics['reciprocal_rank'],
                hit_rate_at_k=metrics['hit_rate_at_k']
            )
            results.append(result)
        
        return results
    
    def _calculate_metrics(self, retrieved: List[Tuple[Chunk, float]], 
                          target_doc: str, k_values: List[int]) -> Dict[str, Any]:
        """Calculate evaluation metrics"""
        metrics = {
            'precision_at_k': {},
            'recall_at_k': {},
            'hit_rate_at_k': {},
            'reciprocal_rank': 0.0
        }
        
        # Handle case where target_doc is NaN or invalid
        if pd.isna(target_doc) or target_doc == 'nan' or not target_doc:
            # Return zero metrics for invalid target documents
            for k in k_values:
                metrics['precision_at_k'][k] = 0.0
                metrics['recall_at_k'][k] = 0.0
                metrics['hit_rate_at_k'][k] = False
            return metrics
        
        # Find first relevant chunk position
        first_relevant_pos = None
        for i, (chunk, score) in enumerate(retrieved):
            if chunk.doc_id == target_doc:
                if first_relevant_pos is None:
                    first_relevant_pos = i + 1  # 1-indexed
                break
        
        # Calculate reciprocal rank
        if first_relevant_pos is not None:
            metrics['reciprocal_rank'] = 1.0 / first_relevant_pos
        
        # Calculate metrics for each k
        for k in k_values:
            top_k = retrieved[:k]
            relevant_in_top_k = sum(1 for chunk, score in top_k if chunk.doc_id == target_doc)
            
            # Precision@k
            metrics['precision_at_k'][k] = relevant_in_top_k / k if k > 0 else 0
            
            # For recall@k, we assume all chunks from target doc are relevant
            # This is a limitation of this evaluation approach
            total_relevant = 1  # Simplified assumption
            metrics['recall_at_k'][k] = min(relevant_in_top_k / total_relevant, 1.0)
            
            # Hit rate@k
            metrics['hit_rate_at_k'][k] = relevant_in_top_k > 0
        
        return metrics
    
    def compare_strategies(self, strategies: List[ChunkingStrategy]) -> Dict[str, Any]:
        """Compare multiple chunking strategies"""
        results = {}
        
        for strategy in strategies:
            strategy_results = self.evaluate_strategy(strategy)
            results[strategy.name] = strategy_results
        
        # Calculate aggregate metrics
        comparison = self._aggregate_results(results)
        return comparison
    
    def _aggregate_results(self, results: Dict[str, List[EvaluationResult]]) -> Dict[str, Any]:
        """Aggregate results across all questions"""
        aggregated = {}
        
        for strategy_name, strategy_results in results.items():
            if not strategy_results:
                continue
                
            metrics = {
                'mean_reciprocal_rank': np.mean([r.reciprocal_rank for r in strategy_results]),
                'precision_at_k': {},
                'recall_at_k': {},
                'hit_rate_at_k': {},
                'num_questions': len(strategy_results)
            }
            
            for k in self.k_values:
                metrics['precision_at_k'][k] = np.mean([r.precision_at_k[k] for r in strategy_results])
                metrics['recall_at_k'][k] = np.mean([r.recall_at_k[k] for r in strategy_results])
                metrics['hit_rate_at_k'][k] = np.mean([r.hit_rate_at_k[k] for r in strategy_results])
            
            aggregated[strategy_name] = metrics
        
        return aggregated
    
    def print_results(self, comparison: Dict[str, Any]):
        """Print comparison results"""
        print("\n" + "="*80)
        print("CHUNKING STRATEGY COMPARISON RESULTS")
        print("="*80)
        
        for strategy_name, metrics in comparison.items():
            print(f"\nStrategy: {strategy_name}")
            print("-" * 50)
            print(f"Number of questions evaluated: {metrics['num_questions']}")
            print(f"Mean Reciprocal Rank: {metrics['mean_reciprocal_rank']:.4f}")
            
            print("\nPrecision@K:")
            for k in sorted(metrics['precision_at_k'].keys()):
                print(f"  P@{k}: {metrics['precision_at_k'][k]:.4f}")
            
            print("\nRecall@K:")
            for k in sorted(metrics['recall_at_k'].keys()):
                print(f"  R@{k}: {metrics['recall_at_k'][k]:.4f}")
            
            print("\nHit Rate@K:")
            for k in sorted(metrics['hit_rate_at_k'].keys()):
                print(f"  HR@{k}: {metrics['hit_rate_at_k'][k]:.4f}")

def main():
    """Main execution function"""
    # Configuration
    QUESTIONS_FILE = "textwave/qa_resources/question.tsv"  # Update this path
    DOCUMENTS_DIR = "textwave/storage/"  # Update this path - where article files are stored
    
    # Initialize evaluator
    evaluator = ChunkingEvaluator(QUESTIONS_FILE, DOCUMENTS_DIR)
    
    # Debug: Print some information about the setup
    print(f"\nDEBUG INFO:")
    print(f"Questions file: {QUESTIONS_FILE}")
    print(f"Documents directory: {DOCUMENTS_DIR}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if directories exist
    if os.path.exists(QUESTIONS_FILE):
        print(f"✓ Questions file found")
        # Print first few rows to verify
        try:
            df_sample = pd.read_csv(QUESTIONS_FILE, sep='\t', nrows=3)
            print(f"✓ Sample ArticleFile values: {df_sample['ArticleFile'].tolist()}")
        except Exception as e:
            print(f"✗ Error reading questions file: {e}")
    else:
        print(f"✗ Questions file NOT found")
    
    if os.path.exists(DOCUMENTS_DIR):
        print(f"✓ Documents directory found")
        # List some files in the directory
        files = os.listdir(DOCUMENTS_DIR)[:5]  # Show first 5 files
        print(f"✓ Sample files in directory: {files}")
    else:
        print(f"✗ Documents directory NOT found")
    print()
    
    # Define chunking strategies to compare
    strategies = [
        # Fixed-length strategies with different parameters
        FixedLengthChunking(chunk_size=512, overlap=64),
        FixedLengthChunking(chunk_size=1024, overlap=128),
        FixedLengthChunking(chunk_size=2048, overlap=256),
        
        # Sentence-based strategies
        SentenceChunking(sentences_per_chunk=3, min_chunk_size=100),
        SentenceChunking(sentences_per_chunk=5, min_chunk_size=200),
    ]
    
    # Print strategy parameters
    print("CHUNKING STRATEGIES BEING EVALUATED:")
    print("="*50)
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. {strategy.name}")
        if isinstance(strategy, FixedLengthChunking):
            print(f"   - Chunk size: {strategy.chunk_size} characters")
            print(f"   - Overlap: {strategy.overlap} characters")
        elif isinstance(strategy, SentenceChunking):
            print(f"   - Sentences per chunk: {strategy.sentences_per_chunk}")
            print(f"   - Minimum chunk size: {strategy.min_chunk_size} characters")
    
    # Run comparison
    try:
        comparison_results = evaluator.compare_strategies(strategies)
        evaluator.print_results(comparison_results)
        
        # Save results to JSON for further analysis
        with open('chunking_evaluation_results.json', 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: chunking_evaluation_results.json")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Please check that:")
        print("1. The questions file path is correct")
        print("2. The documents directory path is correct")
        print("3. All article files referenced in the TSV exist in the documents directory")

if __name__ == "__main__":
    main()