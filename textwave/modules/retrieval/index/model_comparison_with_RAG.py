import pandas as pd
import numpy as np
import os
import json
import time
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import re
import sys
import nltk

# Import optimal components from previous tasks
sys.path.append('..')
sys.path.append('.')

try:
    from reranker import Reranker
    from hnsw import FaissHNSW
    from mistralai import Mistral
    COMPONENTS_AVAILABLE = True
    print("Successfully imported all components")
except ImportError as e:
    print(f"Could not import components: {e}")
    COMPONENTS_AVAILABLE = False

# For evaluation metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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
class RAGEvaluationResult:
    """Stores RAG evaluation results for a single question"""
    question_id: int
    question: str
    ground_truth: str
    generated_answer: str
    retrieved_context: List[str]
    model_name: str
    difficulty: str
    article_title: str
    bleu_score: float
    rouge_scores: Dict[str, float]
    semantic_similarity: float
    exact_match: bool
    response_time: float
    retrieval_time: float
    
@dataclass
class ModelComparison:
    """Comparison between RAG and baseline performance"""
    model_name: str
    rag_results: Dict[str, float]
    baseline_results: Dict[str, float]
    improvements: Dict[str, float]
    difficulty_breakdown: Dict[str, Dict[str, Any]]

class SentenceChunking:
    """Optimal chunking strategy from Task 1"""
    
    def __init__(self, sentences_per_chunk: int = 5, min_chunk_size: int = 200):
        self.name = f"sentence_{sentences_per_chunk}_{min_chunk_size}"
        self.sentences_per_chunk = sentences_per_chunk
        self.min_chunk_size = min_chunk_size
    
    def preprocess_text(self, text: str) -> str:
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

class OptimalRetriever:
    """Optimal retrieval system from Tasks 1-2: HNSW + sentence_5_200"""
    
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        
        # Disable OpenMP threading to avoid conflicts
        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            max_features=2000  # Conservative for stability
        )
        
        # Build TF-IDF matrix
        corpus = [chunk.text for chunk in chunks]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        dense_vectors = tfidf_matrix.toarray().astype(np.float32)
        
        # Initialize HNSW index with conservative settings
        dim = dense_vectors.shape[1]
        
        try:
            self.index = FaissHNSW(
                dim=dim, 
                M=8,  # Conservative settings
                efConstruction=40,
                efSearch=16,
                metric='cosine',
                normalize=True
            )
            
            metadata = list(range(len(chunks)))
            self.index.add_embeddings(dense_vectors, metadata)
            print(f"Built HNSW retrieval system with {len(chunks)} chunks")
            self.use_hnsw = True
            
        except Exception as e:
            print(f"HNSW failed, using TF-IDF fallback: {e}")
            self.tfidf_matrix = tfidf_matrix
            self.use_hnsw = False
    
    def search(self, query: str, k: int = 20) -> List[Chunk]:
        if self.use_hnsw:
            return self._search_hnsw(query, k)
        else:
            return self._search_tfidf(query, k)
    
    def _search_hnsw(self, query: str, k: int) -> List[Chunk]:
        query_vector = self.vectorizer.transform([query])
        query_dense = query_vector.toarray().astype(np.float32)
        
        if self.index.normalize:
            import faiss
            faiss.normalize_L2(query_dense)
        
        distances, indices = self.index.index.search(query_dense, k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.chunks):
                results.append(self.chunks[idx])
        
        return results
    
    def _search_tfidf(self, query: str, k: int) -> List[Chunk]:
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append(self.chunks[idx])
        
        return results

class OptimalReRanker:
    """Sequential re-ranking pipeline for optimal relevance scoring"""
    
    def __init__(self):
        self.reranker = Reranker(type="sequential")
        print("Initialized sequential re-ranker")
    
    def rerank(self, query: str, chunks: List[Chunk], k: int = 5) -> List[Chunk]:
        if not chunks:
            return []
        
        try:
            context = [chunk.text for chunk in chunks]
            
            # Apply sequential re-ranking (TF-IDF → Cross-Encoder)
            reranked_docs, reranked_indices, reranked_scores = self.reranker.rerank(
                query, context, seq_k1=10, seq_k2=k
            )
            
            # Return re-ranked chunks
            reranked_chunks = []
            for i, orig_idx in enumerate(reranked_indices):
                if orig_idx < len(chunks):
                    reranked_chunks.append(chunks[orig_idx])
            
            return reranked_chunks[:k]
            
        except Exception as e:
            print(f"Re-ranking failed, returning original order: {e}")
            return chunks[:k]

class AnswerEvaluator:
    """Evaluates generated answers against ground truth"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            max_features=5000
        )
        
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        try:
            ref_words = set(reference.lower().split())
            cand_words = set(candidate.lower().split())
            
            if len(cand_words) == 0:
                return 0.0
            
            overlap = len(ref_words.intersection(cand_words))
            return overlap / len(cand_words)
            
        except Exception:
            return 0.0
    
    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        try:
            ref_words = reference.lower().split()
            cand_words = candidate.lower().split()
            
            if len(ref_words) == 0 or len(cand_words) == 0:
                return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
            
            # ROUGE-1
            ref_unigrams = set(ref_words)
            cand_unigrams = set(cand_words)
            rouge_1 = len(ref_unigrams.intersection(cand_unigrams)) / len(ref_unigrams)
            
            # ROUGE-2
            ref_bigrams = set(zip(ref_words[:-1], ref_words[1:]))
            cand_bigrams = set(zip(cand_words[:-1], cand_words[1:]))
            rouge_2 = len(ref_bigrams.intersection(cand_bigrams)) / len(ref_bigrams) if len(ref_bigrams) > 0 else 0.0
            
            # ROUGE-L (approximation)
            rouge_l = rouge_1
            
            return {
                'rouge-1': rouge_1,
                'rouge-2': rouge_2,
                'rouge-l': rouge_l
            }
            
        except Exception:
            return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
    
    def calculate_semantic_similarity(self, reference: str, candidate: str) -> float:
        try:
            docs = [reference, candidate]
            tfidf_matrix = self.vectorizer.fit_transform(docs)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception:
            return 0.0
    
    def calculate_exact_match(self, reference: str, candidate: str) -> bool:
        ref_clean = ' '.join(reference.lower().split())
        cand_clean = ' '.join(candidate.lower().split())
        return ref_clean == cand_clean
    
    def evaluate_answer(self, reference: str, candidate: str) -> Dict[str, Any]:
        return {
            'bleu_score': self.calculate_bleu_score(reference, candidate),
            'rouge_scores': self.calculate_rouge_scores(reference, candidate),
            'semantic_similarity': self.calculate_semantic_similarity(reference, candidate),
            'exact_match': self.calculate_exact_match(reference, candidate)
        }

class RAGMistralEvaluator:
    """Complete RAG-enhanced evaluation system"""
    
    def __init__(self, api_key: str, questions_file: str, documents_dir: str):
        if not COMPONENTS_AVAILABLE:
            raise ImportError("Required components not available")
        
        self.client = Mistral(api_key=api_key)
        self.evaluator = AnswerEvaluator()
        self.questions_file = questions_file
        self.documents_dir = documents_dir
        
        # Mistral models to evaluate
        self.models = [
            "mistral-small-latest",
            "mistral-medium-latest",
            "mistral-large-latest"
        ]
        
        # Load questions
        self.questions_df = pd.read_csv(questions_file, sep='\t')
        self.valid_questions = self.questions_df.dropna(subset=['Question', 'Answer', 'DifficultyFromAnswerer'])
        print(f"Loaded {len(self.valid_questions)} valid questions")
        
        # Initialize RAG system with optimal strategies
        self.setup_rag_system()
    
    def load_document(self, filename: str) -> str:
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
    
    def setup_rag_system(self):
        """Initialize RAG system with optimal strategies"""
        print("Setting up RAG system with optimal strategies...")
        
        # 1. Optimal chunking: sentence_5_200
        chunker = SentenceChunking(sentences_per_chunk=5, min_chunk_size=200)
        
        # 2. Create chunks from documents
        all_chunks = []
        unique_docs = self.valid_questions['ArticleFile'].unique()
        
        for doc_file in unique_docs:
            doc_text = self.load_document(doc_file)
            if doc_text:
                chunks = chunker.chunk_document(doc_text, doc_file)
                all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks from {len(unique_docs)} documents")
        
        # 3. Optimal retrieval: HNSW indexing
        self.retriever = OptimalRetriever(all_chunks)
        
        # 4. Optimal re-ranking: Sequential re-ranking
        try:
            self.reranker = OptimalReRanker()
        except Exception as e:
            print(f"Re-ranker failed to initialize: {e}")
            self.reranker = None
        
        print("RAG system setup complete")
    
    def generate_rag_answer(self, model_name: str, question: str, max_retries: int = 3) -> Tuple[str, List[str], float, float]:
        """Generate answer using RAG-enhanced approach"""
        
        # 1. Retrieve relevant chunks
        retrieval_start = time.time()
        retrieved_chunks = self.retriever.search(question, k=20)
        
        # 2. Re-rank chunks if available
        if self.reranker:
            try:
                retrieved_chunks = self.reranker.rerank(question, retrieved_chunks, k=5)
            except Exception as e:
                print(f"Re-ranking failed: {e}")
                retrieved_chunks = retrieved_chunks[:5]
        else:
            retrieved_chunks = retrieved_chunks[:5]
        
        retrieval_time = time.time() - retrieval_start
        
        # 3. Create context from retrieved chunks
        context_texts = [chunk.text for chunk in retrieved_chunks]
        context_str = '\n\n'.join(context_texts[:3])  # Use top 3 chunks
        
        # 4. Create RAG-enhanced prompt
        prompt = f"""Answer the following question using the provided context. Provide a clear, concise, and factual answer.

Context: {context_str}

Question: {question}

Answer:"""
        
        # 5. Generate answer with Mistral
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = self.client.chat.complete(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=200,
                    temperature=0.1
                )
                
                response_time = time.time() - start_time
                
                if response.choices and len(response.choices) > 0:
                    answer = response.choices[0].message.content.strip()
                    return answer, context_texts, response_time, retrieval_time
                
            except Exception as e:
                print(f"Error with {model_name} on attempt {attempt + 1}: {e}")
                if "429" in str(e) or "capacity exceeded" in str(e):
                    wait_time = 20 + (attempt * 10)
                    print(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                elif attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return "No answer generated", [], 0.0, retrieval_time
    
    def evaluate_rag_model(self, model_name: str, max_questions: Optional[int] = None) -> List[RAGEvaluationResult]:
        """Evaluate a single model with RAG enhancement"""
        print(f"\nEvaluating {model_name} with RAG enhancement...")
        
        results = []
        questions_to_process = self.valid_questions.head(max_questions) if max_questions else self.valid_questions
        
        for idx, row in questions_to_process.iterrows():
            if len(results) % 20 == 0:
                print(f"  Progress: {len(results)}/{len(questions_to_process)} questions")
            
            question = str(row['Question'])
            ground_truth = str(row['Answer'])
            difficulty = str(row['DifficultyFromAnswerer']).lower()
            article_title = str(row.get('ArticleTitle', 'Unknown'))
            
            # Generate RAG-enhanced answer
            generated_answer, context, response_time, retrieval_time = self.generate_rag_answer(model_name, question)
            
            # Evaluate answer quality
            eval_metrics = self.evaluator.evaluate_answer(ground_truth, generated_answer)
            
            # Create result
            result = RAGEvaluationResult(
                question_id=idx,
                question=question,
                ground_truth=ground_truth,
                generated_answer=generated_answer,
                retrieved_context=context,
                model_name=model_name,
                difficulty=difficulty,
                article_title=article_title,
                bleu_score=eval_metrics['bleu_score'],
                rouge_scores=eval_metrics['rouge_scores'],
                semantic_similarity=eval_metrics['semantic_similarity'],
                exact_match=eval_metrics['exact_match'],
                response_time=response_time,
                retrieval_time=retrieval_time
            )
            
            results.append(result)
            
            # Rate limiting
            time.sleep(20.0)
        
        print(f"Completed RAG evaluation of {model_name}: {len(results)} questions")
        return results
    
    def load_baseline_results(self) -> Dict[str, Any]:
        """Load baseline results from Task 4"""
        try:
            with open('mistral_baseline_summary_results.json', 'r') as f:
                baseline_data = json.load(f)
            print("Loaded baseline results from Task 4")
            return baseline_data
        except FileNotFoundError:
            print("Baseline results not found. Please run Task 4 first.")
            return {}
    
    def compare_with_baseline(self, rag_results: Dict[str, List[RAGEvaluationResult]]) -> Dict[str, ModelComparison]:
        """Compare RAG results with baseline performance"""
        baseline_data = self.load_baseline_results()
        
        if not baseline_data:
            print("Cannot perform baseline comparison without Task 4 results")
            return {}
        
        comparisons = {}
        
        for model_name, rag_result_list in rag_results.items():
            if model_name not in baseline_data:
                continue
            
            baseline = baseline_data[model_name]
            
            # Aggregate RAG results
            rag_metrics = self._aggregate_rag_results(rag_result_list)
            
            # Calculate improvements
            improvements = {
                'bleu_improvement': rag_metrics['avg_bleu'] - baseline['avg_bleu'],
                'rouge1_improvement': rag_metrics['avg_rouge_1'] - baseline['avg_rouge_1'],
                'semantic_improvement': rag_metrics['avg_semantic_similarity'] - baseline['avg_semantic_similarity'],
                'exact_match_improvement': rag_metrics['exact_match_rate'] - baseline['exact_match_rate'],
                'bleu_relative_improvement': ((rag_metrics['avg_bleu'] - baseline['avg_bleu']) / baseline['avg_bleu'] * 100) if baseline['avg_bleu'] > 0 else 0,
                'rouge1_relative_improvement': ((rag_metrics['avg_rouge_1'] - baseline['avg_rouge_1']) / baseline['avg_rouge_1'] * 100) if baseline['avg_rouge_1'] > 0 else 0
            }
            
            # Difficulty-stratified comparison
            difficulty_breakdown = self._compare_difficulty_breakdown(rag_result_list, baseline.get('difficulty_breakdown', {}))
            
            comparisons[model_name] = ModelComparison(
                model_name=model_name,
                rag_results=rag_metrics,
                baseline_results=baseline,
                improvements=improvements,
                difficulty_breakdown=difficulty_breakdown
            )
        
        return comparisons
    
    def _aggregate_rag_results(self, results: List[RAGEvaluationResult]) -> Dict[str, float]:
        """Aggregate RAG evaluation results"""
        if not results:
            return {}
        
        return {
            'total_questions': len(results),
            'avg_bleu': np.mean([r.bleu_score for r in results]),
            'avg_rouge_1': np.mean([r.rouge_scores['rouge-1'] for r in results]),
            'avg_rouge_2': np.mean([r.rouge_scores['rouge-2'] for r in results]),
            'avg_rouge_l': np.mean([r.rouge_scores['rouge-l'] for r in results]),
            'avg_semantic_similarity': np.mean([r.semantic_similarity for r in results]),
            'exact_match_rate': np.mean([r.exact_match for r in results]),
            'avg_response_time': np.mean([r.response_time for r in results]),
            'avg_retrieval_time': np.mean([r.retrieval_time for r in results])
        }
    
    def _compare_difficulty_breakdown(self, rag_results: List[RAGEvaluationResult], baseline_breakdown: Dict) -> Dict[str, Any]:
        """Compare difficulty-stratified results"""
        difficulties = set([r.difficulty for r in rag_results])
        comparison = {}
        
        for diff in difficulties:
            diff_results = [r for r in rag_results if r.difficulty == diff]
            if not diff_results:
                continue
            
            rag_metrics = {
                'avg_bleu': np.mean([r.bleu_score for r in diff_results]),
                'avg_rouge_1': np.mean([r.rouge_scores['rouge-1'] for r in diff_results]),
                'avg_semantic_similarity': np.mean([r.semantic_similarity for r in diff_results]),
                'exact_match_rate': np.mean([r.exact_match for r in diff_results]),
            }
            
            baseline_metrics = baseline_breakdown.get(diff, {})
            
            improvements = {}
            if baseline_metrics:
                improvements = {
                    'bleu_improvement': rag_metrics['avg_bleu'] - baseline_metrics.get('avg_bleu', 0),
                    'rouge1_improvement': rag_metrics['avg_rouge_1'] - baseline_metrics.get('avg_rouge_1', 0),
                    'semantic_improvement': rag_metrics['avg_semantic_similarity'] - baseline_metrics.get('avg_semantic_similarity', 0),
                    'exact_match_improvement': rag_metrics['exact_match_rate'] - baseline_metrics.get('exact_match_rate', 0)
                }
            
            comparison[diff] = {
                'rag_metrics': rag_metrics,
                'baseline_metrics': baseline_metrics,
                'improvements': improvements,
                'count': len(diff_results)
            }
        
        return comparison
    
    def run_full_evaluation(self, max_questions_per_model: Optional[int] = None) -> Dict[str, ModelComparison]:
        """Run complete RAG evaluation and comparison"""
        print("="*80)
        print("RAG-ENHANCED MISTRAL EVALUATION")
        print("Optimal Strategies: sentence_5_200 + HNSW + Sequential Re-ranking")
        print("="*80)
        
        # Evaluate all models with RAG
        rag_results = {}
        for model_name in self.models:
            try:
                results = self.evaluate_rag_model(model_name, max_questions_per_model)
                rag_results[model_name] = results
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
        
        # Compare with baseline
        comparisons = self.compare_with_baseline(rag_results)
        
        # Save detailed results
        self._save_detailed_results(rag_results)
        
        return comparisons
    
    def _save_detailed_results(self, rag_results: Dict[str, List[RAGEvaluationResult]]):
        """Save detailed RAG evaluation results"""
        detailed_results = {}
        
        for model_name, results in rag_results.items():
            detailed_results[model_name] = []
            
            for result in results:
                detailed_results[model_name].append({
                    'question_id': result.question_id,
                    'question': result.question,
                    'ground_truth': result.ground_truth,
                    'generated_answer': result.generated_answer,
                    'retrieved_context': result.retrieved_context,
                    'difficulty': result.difficulty,
                    'bleu_score': result.bleu_score,
                    'rouge_scores': result.rouge_scores,
                    'semantic_similarity': result.semantic_similarity,
                    'exact_match': result.exact_match,
                    'response_time': result.response_time,
                    'retrieval_time': result.retrieval_time
                })
        
        with open('rag_enhanced_detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print("Detailed RAG results saved to: rag_enhanced_detailed_results.json")
    
    def print_comparison_results(self, comparisons: Dict[str, ModelComparison]):
        """Print comprehensive comparison results"""
        if not comparisons:
            print("No comparison results available")
            return
        
        print("\n" + "="*120)
        print("RAG vs BASELINE COMPARISON RESULTS")
        print("="*120)
        
        # Overall comparison table
        print(f"\n{'Model':<20} {'System':<10} {'BLEU':<8} {'ROUGE-1':<9} {'Semantic':<9} {'Exact Match':<11} {'Avg Time':<10}")
        print("-" * 110)
        
        for model_name, comp in comparisons.items():
            # Baseline row
            print(f"{model_name:<20} {'Baseline':<10} {comp.baseline_results['avg_bleu']:<8.3f} "
                  f"{comp.baseline_results['avg_rouge_1']:<9.3f} {comp.baseline_results['avg_semantic_similarity']:<9.3f} "
                  f"{comp.baseline_results['exact_match_rate']:<11.3f} {comp.baseline_results['avg_response_time']:<10.2f}s")
            
            # RAG row
            print(f"{'':<20} {'RAG':<10} {comp.rag_results['avg_bleu']:<8.3f} "
                  f"{comp.rag_results['avg_rouge_1']:<9.3f} {comp.rag_results['avg_semantic_similarity']:<9.3f} "
                  f"{comp.rag_results['exact_match_rate']:<11.3f} {comp.rag_results['avg_response_time']:<10.2f}s")
            
            # Improvement row
            print(f"{'':<20} {'Δ (abs)':<10} {comp.improvements['bleu_improvement']:<8.3f} "
                  f"{comp.improvements['rouge1_improvement']:<9.3f} {comp.improvements['semantic_improvement']:<9.3f} "
                  f"{comp.improvements['exact_match_improvement']:<11.3f} {'':<10}")
            
            print("-" * 110)
        
        # Improvement summary
        print(f"\n" + "="*80)
        print("IMPROVEMENT ANALYSIS")
        print("="*80)
        
        for model_name, comp in comparisons.items():
            print(f"\n{model_name}:")
            print(f"  BLEU improvement: {comp.improvements['bleu_improvement']:+.3f} ({comp.improvements['bleu_relative_improvement']:+.1f}%)")
            print(f"  ROUGE-1 improvement: {comp.improvements['rouge1_improvement']:+.3f} ({comp.improvements['rouge1_relative_improvement']:+.1f}%)")
            print(f"  Semantic similarity improvement: {comp.improvements['semantic_improvement']:+.3f}")
            print(f"  Exact match improvement: {comp.improvements['exact_match_improvement']:+.3f}")
        
        # Difficulty-stratified improvements
        print(f"\n" + "="*80)
        print("DIFFICULTY-STRATIFIED IMPROVEMENTS")
        print("="*80)
        
        for model_name, comp in comparisons.items():
            print(f"\n{model_name} - Difficulty Breakdown:")
            print("-" * 50)
            
            for diff, data in comp.difficulty_breakdown.items():
                if not data.get('improvements'):
                    continue
                    
                print(f"  {diff.upper()} ({data['count']} questions):")
                print(f"    BLEU: {data['improvements']['bleu_improvement']:+.3f}")
                print(f"    ROUGE-1: {data['improvements']['rouge1_improvement']:+.3f}")
                print(f"    Semantic: {data['improvements']['semantic_improvement']:+.3f}")
                print(f"    Exact Match: {data['improvements']['exact_match_improvement']:+.3f}")
        
        # Overall conclusions
        print(f"\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)
        
        if comparisons:
            # Calculate average improvements across models
            avg_bleu_imp = np.mean([comp.improvements['bleu_improvement'] for comp in comparisons.values()])
            avg_rouge_imp = np.mean([comp.improvements['rouge1_improvement'] for comp in comparisons.values()])
            avg_semantic_imp = np.mean([comp.improvements['semantic_improvement'] for comp in comparisons.values()])
            avg_exact_imp = np.mean([comp.improvements['exact_match_improvement'] for comp in comparisons.values()])
            
            print(f"Average improvements across all models:")
            print(f"  BLEU Score: {avg_bleu_imp:+.3f}")
            print(f"  ROUGE-1 Score: {avg_rouge_imp:+.3f}")
            print(f"  Semantic Similarity: {avg_semantic_imp:+.3f}")
            print(f"  Exact Match Rate: {avg_exact_imp:+.3f}")
            
            # Identify best performing RAG model
            best_model = max(comparisons.keys(), key=lambda m: comparisons[m].rag_results['avg_bleu'])
            print(f"\nBest RAG Performance: {best_model}")
            
            # Determine if RAG provides consistent improvements
            consistent_improvements = []
            if avg_bleu_imp > 0:
                consistent_improvements.append("BLEU")
            if avg_rouge_imp > 0:
                consistent_improvements.append("ROUGE-1")
            if avg_semantic_imp > 0:
                consistent_improvements.append("Semantic Similarity")
            if avg_exact_imp > 0:
                consistent_improvements.append("Exact Match")
            
            if consistent_improvements:
                print(f"RAG shows consistent improvements in: {', '.join(consistent_improvements)}")
            else:
                print("RAG shows mixed results compared to baseline")
        
        # Save comparison results
        comparison_summary = {}
        for model_name, comp in comparisons.items():
            comparison_summary[model_name] = {
                'rag_results': comp.rag_results,
                'baseline_results': comp.baseline_results,
                'improvements': comp.improvements,
                'difficulty_breakdown': comp.difficulty_breakdown
            }
        
        with open('rag_vs_baseline_comparison.json', 'w') as f:
            json.dump(comparison_summary, f, indent=2)
        
        print(f"\nComparison results saved to: rag_vs_baseline_comparison.json")

def main():
    """Main execution function"""
    # Configuration
    QUESTIONS_FILE = "../../../qa_resources/question.tsv"
    DOCUMENTS_DIR = "../../../storage/"
    
    # Get API key
    API_KEY = os.environ.get('MISTRAL_API_KEY')
    if not API_KEY:
        print("ERROR: MISTRAL_API_KEY environment variable not set")
        print("Please set your Mistral API key:")
        print("export MISTRAL_API_KEY='your-api-key-here'")
        return
    
    print("RAG-Enhanced Mistral Evaluation")
    print("Current working directory:", os.getcwd())
    print(f"Questions file: {QUESTIONS_FILE}")
    print(f"Documents directory: {DOCUMENTS_DIR}")
    print(f"Using API key: {'*' * (len(API_KEY) - 4) + API_KEY[-4:]}")
    print()
    
    if not os.path.exists(QUESTIONS_FILE):
        print(f"ERROR: Questions file not found at {QUESTIONS_FILE}")
        return
    
    if not os.path.exists(DOCUMENTS_DIR):
        print(f"ERROR: Documents directory not found at {DOCUMENTS_DIR}")
        return
    
    if not COMPONENTS_AVAILABLE:
        print("ERROR: Required components not available")
        return
    
    try:
        # Initialize RAG evaluator
        evaluator = RAGMistralEvaluator(API_KEY, QUESTIONS_FILE, DOCUMENTS_DIR)
        
        # Run evaluation (limit questions for testing - remove for full evaluation)
        comparisons = evaluator.run_full_evaluation()  # Remove for full evaluation
        
        # Print comprehensive results
        evaluator.print_comparison_results(comparisons)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()