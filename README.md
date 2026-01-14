# TextWave: RAG System with Comparative Retrieval Analysis

> Retrieval-Augmented Generation system demonstrating that RAG enables smaller, cheaper models to match or exceed larger model performance—Small+RAG (0.156 similarity) outperforms Large baseline (0.149) at lower cost.

## The Problem

Large Language Models face a fundamental limitation: their knowledge is frozen at training time. For domain-specific question answering, this creates three issues:

1. **Stale knowledge**: The model can't answer questions about information after its training cutoff
2. **Hallucination**: Without grounding, models confidently generate plausible but incorrect answers
3. **Cost scaling**: Larger models with more parameters are expensive to run—but are they necessary?

Retrieval-Augmented Generation (RAG) addresses all three by retrieving relevant documents at query time and grounding responses in retrieved context. This project systematically evaluates the components of a RAG pipeline to identify optimal configurations.

## Key Findings

### RAG Makes Small Models Competitive

The most significant finding: RAG enhancement allows smaller models to match or exceed larger model performance:

| Model | System | Semantic Similarity | Cost Implication |
|-------|--------|--------------------:|------------------|
| Mistral-Small | Baseline | 0.112 | Low cost |
| **Mistral-Small** | **RAG** | **0.156** | Low cost + retrieval |
| Mistral-Large | Baseline | 0.149 | High cost |

**Strategic Insight**: Small+RAG (0.156) outperforms Large baseline (0.149) by 4.7% at significantly lower inference cost. This suggests RAG is more cost-effective than model scaling for domain-specific QA.

### Chunking Strategy Comparison

Document chunking significantly impacts retrieval quality:

| Strategy | MRR | P@1 | Hit Rate@10 |
|----------|----:|----:|------------:|
| **Sentence (5 sent, 200 chars)** | **0.924** | **0.897** | 0.968 |
| Fixed (512 chars, 64 overlap) | 0.911 | 0.881 | 0.969 |
| Fixed (1024 chars, 128 overlap) | 0.911 | 0.883 | 0.967 |

**Strategic Decision**: Sentence-based chunking preserves semantic coherence better than arbitrary character boundaries. The 1.3% MRR improvement is consistent across query types.

### Indexing Strategy Trade-offs

| Strategy | Search Time | MRR | P@1 | Best For |
|----------|------------:|----:|----:|----------|
| FAISS Brute Force | 0.613ms | **0.905** | **0.874** | Maximum accuracy |
| **FAISS HNSW** | **0.428ms** | 0.902 | 0.870 | Balanced (recommended) |
| FAISS LSH | 0.352ms | 0.645 | 0.550 | Memory-constrained |

**Trade-off Analysis**: HNSW provides 30% faster search with only 0.3% MRR degradation—an acceptable trade-off for most applications. LSH's 26% MRR drop makes it unsuitable except under extreme memory constraints.

### Reranking Effectiveness

Initial retrieval returns candidates; reranking improves precision:

| Strategy | MRR | P@1 | Latency | ROI |
|----------|----:|----:|--------:|-----|
| Baseline (no rerank) | 0.844 | 0.807 | 0ms | — |
| TF-IDF | 0.858 | 0.822 | 1.70ms | High |
| Sequential (TF-IDF → Cross-Encoder) | 0.893 | 0.879 | 20.87ms | Medium |
| **Cross-Encoder** | **0.897** | **0.881** | 40.25ms | Quality-focused |

**Strategic Decision**: For latency-sensitive applications, TF-IDF reranking provides 1.4% MRR improvement at minimal cost. Cross-encoder reranking is justified when answer quality is paramount.

## System Architecture

```
textwave/
├── Dockerfile
├── CaseStudy.md                    # Detailed performance analysis
├── quality_testing.py
└── textwave/
    ├── app.py                      # Flask API server
    ├── storage/                    # Document corpus
    ├── qa_resources/               # Evaluation datasets
    └── modules/
        ├── extraction/
        │   ├── embedding.py        # Sentence transformer embeddings
        │   └── preprocessing.py    # Chunking strategies
        ├── retrieval/
        │   ├── search.py           # FAISS search interface
        │   ├── reranker.py         # TF-IDF, Cross-encoder, BoW
        │   └── index/
        │       ├── bruteforce.py   # Exact nearest neighbor
        │       ├── hnsw.py         # Approximate (HNSW)
        │       └── lsh.py          # Locality-Sensitive Hashing
        ├── generator/
        │   └── question_answering.py  # Mistral API integration
        └── utils/
            ├── bow.py, tfidf.py    # Reranking utilities
            └── metrics.py          # Evaluation metrics
```

## API Reference

### POST /generate
Generates an answer using the full RAG pipeline: retrieve → rerank → generate.

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the health benefits of green tea?"}'
```

**Response**:
```json
{
  "query": "What are the health benefits of green tea?",
  "answer": "Green tea contains antioxidants that may help...",
  "num_documents_retrieved": 20,
  "num_documents_used": 5
}
```

### GET /health
Health check endpoint.

## Quick Start

### Local Installation

```bash
git clone https://github.com/bruce2tech/textwave.git
cd textwave

pip install -r requirements.txt

export MISTRAL_API_KEY='your-api-key-here'

python textwave/app.py
```

### Docker Deployment

```bash
docker build -t textwave .
docker run -d -p 5000:5000 -e MISTRAL_API_KEY='your-api-key-here' textwave
```

## Configuration

Default parameters in `textwave/app.py`:

```python
CHUNKING_STRATEGY = 'fixed_length'
CHUNKING_PARAMETERS = {"chunk_size": 500, "overlap_size": 50}
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
INDEX_STRATEGY = "bruteforce"
RERANKING_STRATEGY = "tfidf"
```

## Evaluation Suite

The repository includes scripts for systematic component evaluation. JSON result files are generated by running these scripts (gitignored due to size).

| Script | Output | Purpose |
|--------|--------|---------|
| `index_strategy_comparison.py` | `faiss_indexing_comparison_results.json` | Index performance benchmarks |
| `reranker_comparison.py` | `reranking_results.json` | Reranking strategy analysis |
| `mistral_baseline_evaluation.py` | `mistral_baseline_*_results.json` | LLM baseline performance |
| `model_comparison_with_RAG.py` | `rag_*_results.json` | RAG vs baseline comparison |

For detailed analysis, see [CaseStudy.md](CaseStudy.md).

## Technical Insights

### Key Observations

1. **Retrieval quality bounds generation quality**: Even the best LLM cannot recover from poor retrieval. Investment in chunking and reranking yields higher returns than model upgrades.

2. **Semantic chunking outperforms arbitrary boundaries**: Sentence-based chunking preserves meaning units. Fixed-length chunks often split mid-sentence, fragmenting context.

3. **Two-stage reranking balances cost and quality**: Fast TF-IDF filtering followed by expensive cross-encoder scoring on top candidates achieves 94% of pure cross-encoder quality at 50% latency.

### Production Considerations

For deployment at scale:

- **Embedding caching**: Pre-compute and store document embeddings. Re-embedding on every query is wasteful.
- **Index persistence**: FAISS indices should be serialized to disk. Rebuilding on startup delays service availability.
- **Streaming responses**: For long answers, stream tokens to reduce perceived latency.
- **Retrieval logging**: Log retrieved documents alongside answers for debugging and quality monitoring.

### Known Limitations

- Single embedding model (all-MiniLM-L6-v2); domain-specific fine-tuning may improve retrieval
- No query expansion or reformulation; complex queries may benefit from decomposition
- Static corpus; production systems need incremental index updates

## Technologies

- **Backend**: Flask
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Search**: FAISS (Brute-Force, HNSW, LSH)
- **LLM**: Mistral AI API
- **Reranking**: Cross-encoders, TF-IDF, BoW
- **Evaluation**: scikit-learn, qa-metrics

## Requirements

- Python 3.11+
- See `requirements.txt` for full dependencies

## Author

Patrick Bruce

## License

This project is for educational and portfolio purposes.
