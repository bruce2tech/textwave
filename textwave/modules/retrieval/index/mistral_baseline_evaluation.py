import pandas as pd
import numpy as np
import os
import json
import time
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import re

# For Mistral API
try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
    print("Successfully imported Mistral client")
except ImportError as e:
    print(f"Could not import Mistral client: {e}")
    print("Please install: pip install mistralai")
    MISTRAL_AVAILABLE = False

# For answer evaluation metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class EvaluationResult:
    """Stores evaluation results for a single question"""
    question_id: int
    question: str
    ground_truth: str
    generated_answer: str
    model_name: str
    difficulty: str
    article_title: str
    bleu_score: float
    rouge_scores: Dict[str, float]
    semantic_similarity: float
    exact_match: bool
    response_time: float
    
@dataclass
class ModelResults:
    """Aggregated results for a model"""
    model_name: str
    total_questions: int
    avg_bleu: float
    avg_rouge_1: float
    avg_rouge_2: float
    avg_rouge_l: float
    avg_semantic_similarity: float
    exact_match_rate: float
    avg_response_time: float
    difficulty_breakdown: Dict[str, Dict[str, float]]

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
        """Calculate BLEU score between reference and candidate"""
        try:
            # Simple BLEU-1 approximation using word overlap
            ref_words = set(reference.lower().split())
            cand_words = set(candidate.lower().split())
            
            if len(cand_words) == 0:
                return 0.0
            
            overlap = len(ref_words.intersection(cand_words))
            return overlap / len(cand_words)
            
        except Exception:
            return 0.0
    
    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores (simplified implementation)"""
        try:
            ref_words = reference.lower().split()
            cand_words = candidate.lower().split()
            
            if len(ref_words) == 0 or len(cand_words) == 0:
                return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
            
            # ROUGE-1 (unigram overlap)
            ref_unigrams = set(ref_words)
            cand_unigrams = set(cand_words)
            rouge_1 = len(ref_unigrams.intersection(cand_unigrams)) / len(ref_unigrams)
            
            # ROUGE-2 (bigram overlap)
            ref_bigrams = set(zip(ref_words[:-1], ref_words[1:]))
            cand_bigrams = set(zip(cand_words[:-1], cand_words[1:]))
            rouge_2 = len(ref_bigrams.intersection(cand_bigrams)) / len(ref_bigrams) if len(ref_bigrams) > 0 else 0.0
            
            # ROUGE-L (longest common subsequence - simplified)
            rouge_l = rouge_1  # Approximation
            
            return {
                'rouge-1': rouge_1,
                'rouge-2': rouge_2,
                'rouge-l': rouge_l
            }
            
        except Exception:
            return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
    
    def calculate_semantic_similarity(self, reference: str, candidate: str) -> float:
        """Calculate semantic similarity using TF-IDF cosine similarity"""
        try:
            docs = [reference, candidate]
            tfidf_matrix = self.vectorizer.fit_transform(docs)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception:
            return 0.0
    
    def calculate_exact_match(self, reference: str, candidate: str) -> bool:
        """Check for exact match (case-insensitive, whitespace normalized)"""
        ref_clean = ' '.join(reference.lower().split())
        cand_clean = ' '.join(candidate.lower().split())
        return ref_clean == cand_clean
    
    def evaluate_answer(self, reference: str, candidate: str) -> Dict[str, Any]:
        """Comprehensive answer evaluation"""
        return {
            'bleu_score': self.calculate_bleu_score(reference, candidate),
            'rouge_scores': self.calculate_rouge_scores(reference, candidate),
            'semantic_similarity': self.calculate_semantic_similarity(reference, candidate),
            'exact_match': self.calculate_exact_match(reference, candidate)
        }

class MistralEvaluator:
    """Evaluates Mistral models on Q&A tasks"""
    
    def __init__(self, api_key: str, questions_file: str):
        if not MISTRAL_AVAILABLE:
            raise ImportError("Mistral client not available")
        
        self.client = Mistral(api_key=api_key)
        self.evaluator = AnswerEvaluator()
        self.questions_file = questions_file
        
        # Mistral models to evaluate
        self.models = [
            "mistral-small-latest",
            "mistral-medium-latest", 
            "mistral-large-latest"
        ]
        
        # Load questions
        self.questions_df = pd.read_csv(questions_file, sep='\t')
        print(f"Loaded {len(self.questions_df)} questions")
        
        # Filter valid questions
        self.valid_questions = self.questions_df.dropna(subset=['Question', 'Answer', 'DifficultyFromAnswerer'])
        print(f"Found {len(self.valid_questions)} valid questions")
        
        # Show difficulty distribution
        difficulty_counts = self.valid_questions['DifficultyFromAnswerer'].value_counts()
        print(f"Difficulty distribution: {difficulty_counts.to_dict()}")
    
    def generate_answer(self, model_name: str, question: str, max_retries: int = 3) -> Tuple[str, float]:
        """Generate answer using Mistral model"""
        
        # Create a focused prompt for Q&A
        prompt = f"""Please answer the following question clearly and concisely. Provide a direct, factual answer without unnecessary elaboration.

Question: {question}

Answer:"""
        
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
                    max_tokens=200,  # Keep answers concise
                    temperature=0.1   # Low temperature for consistent, factual answers
                )
                
                response_time = time.time() - start_time
                
                if response.choices and len(response.choices) > 0:
                    answer = response.choices[0].message.content.strip()
                    return answer, response_time
                else:
                    print(f"Empty response from {model_name} on attempt {attempt + 1}")
                    
            except Exception as e:
                print(f"Error with {model_name} on attempt {attempt + 1}: {e}")
                if "429" in str(e) or "capacity exceeded" in str(e):
                    # More aggressive backoff for rate limiting
                    wait_time = (2 ** attempt) * 5  # 5, 10, 20 seconds
                    print(f"Rate limited. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                elif attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Regular exponential backoff
                
        return "No answer generated", 0.0
    
    def evaluate_model(self, model_name: str, max_questions: Optional[int] = None) -> List[EvaluationResult]:
        """Evaluate a single Mistral model"""
        print(f"\nEvaluating {model_name}...")
        
        results = []
        questions_to_process = self.valid_questions.head(max_questions) if max_questions else self.valid_questions
        
        for idx, row in questions_to_process.iterrows():
            if len(results) % 20 == 0:
                print(f"  Progress: {len(results)}/{len(questions_to_process)} questions")
            
            question = str(row['Question'])
            ground_truth = str(row['Answer'])
            difficulty = str(row['DifficultyFromAnswerer']).lower()
            article_title = str(row.get('ArticleTitle', 'Unknown'))
            
            # Generate answer
            generated_answer, response_time = self.generate_answer(model_name, question)
            
            # Evaluate answer quality
            eval_metrics = self.evaluator.evaluate_answer(ground_truth, generated_answer)
            
            # Create result
            result = EvaluationResult(
                question_id=idx,
                question=question,
                ground_truth=ground_truth,
                generated_answer=generated_answer,
                model_name=model_name,
                difficulty=difficulty,
                article_title=article_title,
                bleu_score=eval_metrics['bleu_score'],
                rouge_scores=eval_metrics['rouge_scores'],
                semantic_similarity=eval_metrics['semantic_similarity'],
                exact_match=eval_metrics['exact_match'],
                response_time=response_time
            )
            
            results.append(result)
            
            # Rate limiting - increase delay to avoid 429 errors
            time.sleep(2.0)  # Increased from 0.5 to 2.0 seconds
        
        print(f"Completed evaluation of {model_name}: {len(results)} questions")
        return results
    
    def aggregate_results(self, results: List[EvaluationResult]) -> ModelResults:
        """Aggregate evaluation results for a model"""
        if not results:
            return None
        
        model_name = results[0].model_name
        
        # Overall metrics
        avg_bleu = np.mean([r.bleu_score for r in results])
        avg_rouge_1 = np.mean([r.rouge_scores['rouge-1'] for r in results])
        avg_rouge_2 = np.mean([r.rouge_scores['rouge-2'] for r in results])
        avg_rouge_l = np.mean([r.rouge_scores['rouge-l'] for r in results])
        avg_semantic_similarity = np.mean([r.semantic_similarity for r in results])
        exact_match_rate = np.mean([r.exact_match for r in results])
        avg_response_time = np.mean([r.response_time for r in results])
        
        # Difficulty-stratified results
        difficulty_breakdown = {}
        difficulties = set([r.difficulty for r in results])
        
        for diff in difficulties:
            diff_results = [r for r in results if r.difficulty == diff]
            if diff_results:
                difficulty_breakdown[diff] = {
                    'count': len(diff_results),
                    'avg_bleu': np.mean([r.bleu_score for r in diff_results]),
                    'avg_rouge_1': np.mean([r.rouge_scores['rouge-1'] for r in diff_results]),
                    'avg_rouge_2': np.mean([r.rouge_scores['rouge-2'] for r in diff_results]),
                    'avg_rouge_l': np.mean([r.rouge_scores['rouge-l'] for r in diff_results]),
                    'avg_semantic_similarity': np.mean([r.semantic_similarity for r in diff_results]),
                    'exact_match_rate': np.mean([r.exact_match for r in diff_results]),
                    'avg_response_time': np.mean([r.response_time for r in diff_results])
                }
        
        return ModelResults(
            model_name=model_name,
            total_questions=len(results),
            avg_bleu=avg_bleu,
            avg_rouge_1=avg_rouge_1,
            avg_rouge_2=avg_rouge_2,
            avg_rouge_l=avg_rouge_l,
            avg_semantic_similarity=avg_semantic_similarity,
            exact_match_rate=exact_match_rate,
            avg_response_time=avg_response_time,
            difficulty_breakdown=difficulty_breakdown
        )
    
    def run_full_evaluation(self, max_questions_per_model: Optional[int] = None) -> Dict[str, ModelResults]:
        """Run evaluation on all Mistral models"""
        print("="*80)
        print("MISTRAL MODELS BASELINE EVALUATION")
        print("Stand-alone Q&A (No RAG)")
        print("="*80)
        
        all_results = {}
        all_individual_results = {}
        
        for model_name in self.models:
            try:
                # Evaluate model
                individual_results = self.evaluate_model(model_name, max_questions_per_model)
                
                # Aggregate results
                aggregated = self.aggregate_results(individual_results)
                
                if aggregated:
                    all_results[model_name] = aggregated
                    all_individual_results[model_name] = individual_results
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue
        
        # Save detailed results
        self.save_detailed_results(all_individual_results)
        
        return all_results
    
    def save_detailed_results(self, all_individual_results: Dict[str, List[EvaluationResult]]):
        """Save detailed results to JSON"""
        detailed_results = {}
        
        for model_name, results in all_individual_results.items():
            detailed_results[model_name] = []
            
            for result in results:
                detailed_results[model_name].append({
                    'question_id': result.question_id,
                    'question': result.question,
                    'ground_truth': result.ground_truth,
                    'generated_answer': result.generated_answer,
                    'difficulty': result.difficulty,
                    'article_title': result.article_title,
                    'bleu_score': result.bleu_score,
                    'rouge_scores': result.rouge_scores,
                    'semantic_similarity': result.semantic_similarity,
                    'exact_match': result.exact_match,
                    'response_time': result.response_time
                })
        
        with open('mistral_baseline_detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"Detailed results saved to: mistral_baseline_detailed_results.json")
    
    def print_results(self, results: Dict[str, ModelResults]):
        """Print comprehensive evaluation results"""
        if not results:
            print("No results to display")
            return
        
        print("\n" + "="*100)
        print("MISTRAL MODELS EVALUATION RESULTS")
        print("="*100)
        
        # Overall comparison table
        print(f"\n{'Model':<25} {'Questions':<10} {'BLEU':<8} {'ROUGE-1':<9} {'ROUGE-L':<9} {'Semantic':<9} {'Exact Match':<11} {'Avg Time':<10}")
        print("-" * 95)
        
        for model_name, result in results.items():
            print(f"{model_name:<25} {result.total_questions:<10} {result.avg_bleu:<8.3f} "
                  f"{result.avg_rouge_1:<9.3f} {result.avg_rouge_l:<9.3f} {result.avg_semantic_similarity:<9.3f} "
                  f"{result.exact_match_rate:<11.3f} {result.avg_response_time:<10.2f}s")
        
        # Difficulty-stratified results
        print(f"\n" + "="*60)
        print("DIFFICULTY-STRATIFIED RESULTS")
        print("="*60)
        
        for model_name, result in results.items():
            print(f"\n{model_name}:")
            print("-" * 50)
            
            if not result.difficulty_breakdown:
                print("  No difficulty breakdown available")
                continue
                
            for difficulty, metrics in result.difficulty_breakdown.items():
                print(f"  {difficulty.upper()} ({metrics['count']} questions):")
                print(f"    BLEU: {metrics['avg_bleu']:.3f}")
                print(f"    ROUGE-1: {metrics['avg_rouge_1']:.3f}")
                print(f"    ROUGE-L: {metrics['avg_rouge_l']:.3f}")
                print(f"    Semantic Similarity: {metrics['avg_semantic_similarity']:.3f}")
                print(f"    Exact Match Rate: {metrics['exact_match_rate']:.3f}")
                print(f"    Avg Response Time: {metrics['avg_response_time']:.2f}s")
        
        # Best performing model analysis
        print(f"\n" + "="*60)
        print("PERFORMANCE ANALYSIS")
        print("="*60)
        
        if results:
            best_bleu = max(results.values(), key=lambda x: x.avg_bleu)
            best_rouge = max(results.values(), key=lambda x: x.avg_rouge_1)
            best_semantic = max(results.values(), key=lambda x: x.avg_semantic_similarity)
            best_exact_match = max(results.values(), key=lambda x: x.exact_match_rate)
            fastest_model = min(results.values(), key=lambda x: x.avg_response_time)
            
            print(f"Best BLEU Score: {best_bleu.model_name} ({best_bleu.avg_bleu:.3f})")
            print(f"Best ROUGE-1: {best_rouge.model_name} ({best_rouge.avg_rouge_1:.3f})")
            print(f"Best Semantic Similarity: {best_semantic.model_name} ({best_semantic.avg_semantic_similarity:.3f})")
            print(f"Best Exact Match Rate: {best_exact_match.model_name} ({best_exact_match.exact_match_rate:.3f})")
            print(f"Fastest Response: {fastest_model.model_name} ({fastest_model.avg_response_time:.2f}s)")
        
        # Save aggregated results
        summary_results = {}
        for model_name, result in results.items():
            summary_results[model_name] = {
                'total_questions': result.total_questions,
                'avg_bleu': result.avg_bleu,
                'avg_rouge_1': result.avg_rouge_1,
                'avg_rouge_2': result.avg_rouge_2,
                'avg_rouge_l': result.avg_rouge_l,
                'avg_semantic_similarity': result.avg_semantic_similarity,
                'exact_match_rate': result.exact_match_rate,
                'avg_response_time': result.avg_response_time,
                'difficulty_breakdown': result.difficulty_breakdown
            }
        
        with open('mistral_baseline_summary_results.json', 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        print(f"\nSummary results saved to: mistral_baseline_summary_results.json")

def main():
    """Main execution function"""
    # Configuration
    QUESTIONS_FILE = "../../../qa_resources/question.tsv"  # Adjust path as needed
    
    # You'll need to set your Mistral API key
    API_KEY = os.environ.get('MISTRAL_API_KEY')
    if not API_KEY:
        print("ERROR: MISTRAL_API_KEY environment variable not set")
        print("Please set your Mistral API key:")
        print("export MISTRAL_API_KEY='your-api-key-here'")
        return
    
    print("Mistral Models Baseline Evaluation")
    print("Current working directory:", os.getcwd())
    print(f"Questions file: {QUESTIONS_FILE}")
    print(f"Using API key: {'*' * (len(API_KEY) - 4) + API_KEY[-4:]}")  # Mask API key
    print()
    
    if not os.path.exists(QUESTIONS_FILE):
        print(f"ERROR: Questions file not found at {QUESTIONS_FILE}")
        return
    
    if not MISTRAL_AVAILABLE:
        print("ERROR: Mistral client not available")
        print("Please install: pip install mistralai")
        return
    
    try:
        # Initialize evaluator
        evaluator = MistralEvaluator(API_KEY, QUESTIONS_FILE)
        
        # Run evaluation (limit questions for testing - remove max_questions_per_model for full evaluation)
        results = evaluator.run_full_evaluation()  # Remove this parameter for full evaluation
        
        # Print results
        evaluator.print_results(results)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()