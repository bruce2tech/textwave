# TextWave - Advanced RAG System with Reranking

A sophisticated Retrieval-Augmented Generation (RAG) system that combines semantic search, multiple indexing strategies, and intelligent reranking to provide accurate question answering over document collections.

## Overview

TextWave is a Flask-based RAG system that demonstrates advanced information retrieval and natural language processing techniques. The system processes document collections, creates searchable embeddings, retrieves relevant context, and generates accurate answers using the Mistral AI model.

## Key Features

- **Multiple Indexing Strategies**:
  - Brute-force search for exact nearest neighbor retrieval
  - FAISS HNSW for fast approximate search
  - LSH (Locality-Sensitive Hashing) for scalable similarity search

- **Advanced Reranking**:
  - TF-IDF based reranking
  - Cross-encoder reranking for semantic relevance
  - Bag-of-Words (BoW) reranking
  - Sequential reranking pipeline

- **Flexible Document Processing**:
  - Multiple chunking strategies (fixed-length, sentence-based)
  - Configurable chunk size and overlap
  - Support for various document formats

- **LLM Integration**:
  - Mistral AI for question answering
  - Context-aware response generation
  - Configurable temperature and model selection

- **Comprehensive Evaluation**:
  - Chunking strategy comparison
  - Index performance benchmarking
  - Reranking effectiveness analysis
  - RAG vs. baseline model comparison

## System Architecture

```
textwave/
├── Dockerfile                      # Docker configuration
├── README.md                       # This file
├── CaseStudy.md                    # Comprehensive performance analysis
├── requirements.txt                # Python dependencies
├── quality_testing.py              # Quality assurance testing
├── notebooks/
│   └── demo_generator.ipynb        # Demo notebook
└── textwave/
    ├── app.py                      # Flask API server
    ├── test_qa.py                  # QA testing utilities
    ├── storage/                    # Document corpus
    ├── qa_resources/               # Question-answer datasets
    └── modules/
        ├── extraction/             # Document processing
        │   ├── embedding.py        # Sentence transformer embeddings
        │   └── preprocessing.py    # Chunking strategies
        ├── retrieval/              # Search and reranking
        │   ├── search.py           # FAISS search interface
        │   ├── reranker.py         # Reranking implementations
        │   └── index/              # Indexing strategies
        │       ├── bruteforce.py   # Exact nearest neighbor
        │       ├── hnsw.py         # FAISS HNSW (approximate)
        │       ├── lsh.py          # Locality-Sensitive Hashing
        │       ├── index_strategy_comparison.py
        │       ├── reranker_comparison.py
        │       ├── mistral_baseline_evaluation.py
        │       └── model_comparison_with_RAG.py
        ├── generator/              # Answer generation
        │   └── question_answering.py  # Mistral API integration
        └── utils/                  # Utility functions
            ├── bow.py              # Bag-of-Words
            ├── tfidf.py            # TF-IDF vectorization
            ├── metrics.py          # Evaluation metrics
            └── text_processing.py  # Text utilities
```

## API Endpoints

### POST /generate
Generates an answer to a question using the RAG pipeline. Automatically indexes the corpus on first request, retrieves relevant documents, applies reranking, and generates an answer using Mistral AI.

**Request Body:**
```json
{
  "query": "Your question here"
}
```

**Response:**
```json
{
  "query": "What are the health benefits of green tea?",
  "answer": "Green tea contains antioxidants that may help...",
  "num_documents_retrieved": 20,
  "num_documents_used": 5
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the health benefits of green tea?"}'
```

### GET /health
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy"
}
```

## Installation

```bash
# Clone the repository
git clone https://github.com/bruce2tech/textwave.git
cd textwave

# Install dependencies
pip install -r requirements.txt

# Set up your Mistral API key
export MISTRAL_API_KEY='your-api-key-here'

# Run the Flask server
python textwave/app.py
```

### Docker Deployment

```bash
# Build the Docker image
docker build -t textwave .

# Run the container
docker run -d -p 5000:5000 -e MISTRAL_API_KEY='your-api-key-here' textwave
```

## Configuration

Default system parameters can be configured in `textwave/app.py`:

```python
CHUNKING_STRATEGY = 'fixed_length'
CHUNKING_PARAMETERS = {
    "chunk_size": 500,
    "overlap_size": 50
}
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
INDEX_STRATEGY = "bruteforce"
RERANKING_STRATEGY = "tfidf"
```

## Evaluation and Benchmarking

The repository includes comprehensive evaluation scripts. For detailed analysis results, see [CaseStudy.md](CaseStudy.md).

> **Note**: JSON result files are generated by running the evaluation scripts and are gitignored due to size. Run the corresponding Python scripts to regenerate them.

### Evaluation Scripts

| Script | Output | Description |
|--------|--------|-------------|
| `index_strategy_comparison.py` | `faiss_indexing_comparison_results.json` | FAISS index performance benchmarks |
| `reranker_comparison.py` | `reranking_results.json` | Reranking strategy comparison |
| `mistral_baseline_evaluation.py` | `mistral_baseline_*_results.json` | Baseline LLM evaluation |
| `model_comparison_with_RAG.py` | `rag_*_results.json` | RAG vs baseline comparison |

### Testing Tools
- `quality_testing.py` - Quality assurance testing
- `textwave/test_qa.py` - Question answering testing

## Technologies Used

- **Backend**: Flask
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Search**: FAISS (Brute-Force, HNSW, LSH)
- **LLM**: Mistral AI API
- **Reranking**: Cross-encoders, TF-IDF, BoW
- **Data Processing**: NumPy, scikit-learn

## Performance Results

Based on comprehensive evaluation (see [CaseStudy.md](CaseStudy.md) for detailed analysis):

### Chunking Strategy Comparison

| Strategy | MRR | P@1 | Hit Rate@10 |
|----------|-----|-----|-------------|
| **Sentence (5 sent, 200 chars)** | **0.924** | **0.897** | 0.968 |
| Fixed (512 chars, 64 overlap) | 0.911 | 0.881 | 0.969 |
| Fixed (1024 chars, 128 overlap) | 0.911 | 0.883 | 0.967 |

### Indexing Strategy Comparison

| Strategy | Search Time | MRR | P@1 |
|----------|-------------|-----|-----|
| FAISS Brute Force | 0.613ms | **0.905** | **0.874** |
| **FAISS HNSW** | **0.428ms** | 0.902 | 0.870 |
| FAISS LSH | 0.352ms | 0.645 | 0.550 |

### Reranking Effectiveness

| Strategy | MRR | P@1 | Latency |
|----------|-----|-----|---------|
| **Cross-Encoder** | **0.897** | **0.881** | 40.25ms |
| Sequential (TF-IDF → Cross-Encoder) | 0.893 | 0.879 | 20.87ms |
| TF-IDF | 0.858 | 0.822 | 1.70ms |
| Baseline (no reranking) | 0.844 | 0.807 | 0ms |

### RAG Enhancement Impact

| Model | System | Semantic Similarity | Improvement |
|-------|--------|---------------------|-------------|
| Mistral-Small | Baseline | 0.112 | - |
| Mistral-Small | **RAG** | **0.156** | **+38.4%** |
| Mistral-Large | Baseline | 0.149 | - |

**Key Finding**: RAG makes smaller, cheaper models competitive with larger models. Small+RAG (0.156) outperforms Large baseline (0.149) at lower cost.

## Project Context

This project was developed as part of a graduate course in Creating AI-Enabled Systems at Johns Hopkins University. It demonstrates practical implementation of:
- Retrieval-Augmented Generation (RAG) systems
- Multiple indexing and search strategies
- Reranking algorithms for improved relevance
- LLM integration for question answering
- Comprehensive evaluation methodologies

## Attribution

This repository originated from a course project at Johns Hopkins University. While the course provided initial starter code and project specifications, the majority of the implementation represents significant original work beyond the base requirements.

### Original Contributions (Patrick Bruce):

**Advanced Features & Enhancements:**
- Multi-strategy reranking implementation (TF-IDF, Cross-encoder, BoW, Sequential)
- Index strategy comparison framework
- RAG vs. baseline model evaluation system
- Enhanced question answering module with Mistral integration
- Advanced reranking patches and improvements

**Evaluation & Analysis:**
- Comprehensive chunking strategy evaluation
- FAISS indexing performance comparison
- Reranking effectiveness analysis
- Model comparison with RAG enhancement
- Quality testing framework

**Code Enhancements:**
- Enhanced preprocessing and chunking modules
- Improved retrieval and reranking pipelines
- Utility functions for BoW, TF-IDF, and text processing
- Debugging and testing tools

**Documentation:**
- Comprehensive README documentation
- API usage examples
- Performance analysis and findings

### Course-Provided Base Components:
- Initial project structure
- Base Flask API framework
- Core module interfaces
- Assignment specifications

**Note:** The extensive evaluation suite, advanced reranking implementations, and model comparison tools demonstrate work that significantly extends beyond the original course requirements.

## Author

Patrick Bruce

## License

This project is for educational and portfolio purposes.

## Requirements

See `requirements.txt` for full dependency list. Key requirements:
- Python 3.11+
- Flask
- sentence-transformers
- transformers
- torch
- faiss-cpu
- mistralai
- numpy
- scikit-learn
- nltk
- qa-metrics
