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
├── modules/
│   ├── extraction/          # Document processing and chunking
│   │   └── preprocessing.py
│   ├── retrieval/           # Search and reranking
│   │   ├── index/           # Multiple indexing strategies
│   │   │   ├── bruteforce.py
│   │   │   ├── hnsw.py
│   │   │   └── lsh.py
│   │   ├── search.py
│   │   └── reranker.py      # Reranking implementations
│   ├── generator/           # Answer generation
│   │   └── question_answering.py
│   └── utils/               # Utility functions
│       ├── bow.py
│       ├── tfidf.py
│       └── text_processing.py
└── app.py                   # Flask API server
```

## API Endpoints

### POST /index
Processes and indexes a document collection.

**Parameters:**
- `corpus_directory`: Path to directory containing documents
- `chunking_strategy`: "fixed_length" or "sentence"
- `chunking_parameters`: JSON object with chunk_size and overlap_size
- `embedding_model_name`: Name of the sentence transformer model
- `index_strategy`: "bruteforce", "hnsw", or "lsh"
- `index_parameters`: JSON object with index-specific parameters

**Example:**
```bash
curl -X POST http://localhost:5000/index \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_directory": "storage/documents",
    "chunking_strategy": "fixed_length",
    "chunking_parameters": {"chunk_size": 500, "overlap_size": 50},
    "embedding_model_name": "all-MiniLM-L6-v2",
    "index_strategy": "hnsw"
  }'
```

### POST /query
Queries the indexed documents and generates an answer.

**Parameters:**
- `query`: The question to answer
- `top_k`: Number of chunks to retrieve (default: 5)
- `reranking_strategy`: "tfidf", "cross_encoder", "bow", or "sequential"
- `reranking_parameters`: JSON object with reranking-specific parameters

**Example:**
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main features of the system?",
    "top_k": 5,
    "reranking_strategy": "cross_encoder"
  }'
```

## Installation

```bash
# Clone the repository
git clone https://github.com/bruce2tech/textwave.git
cd textwave

# Install dependencies
pip install flask sentence-transformers faiss-cpu mistralai numpy scikit-learn

# Set up your Mistral API key
export MISTRAL_API_KEY='your-api-key-here'

# Run the Flask server
python -m textwave.app
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

The repository includes comprehensive evaluation scripts:

### Performance Analysis
- `chunking_evaluation_results.json` - Comparison of chunking strategies
- `indexing_comparison_results.json` - Index performance metrics
- `faiss_indexing_comparison_results.json` - FAISS-specific benchmarks

### Reranking Analysis
- `reranking_comparison_results.json` - Reranking strategy comparison
- `advanced_reranking_comparison_results.json` - Advanced reranking metrics
- `textwave/modules/retrieval/index/reranker_comparison.py` - Reranking evaluation script

### Model Comparison
- `textwave/modules/retrieval/index/model_comparison_with_RAG.py` - RAG vs baseline comparison
- `textwave/modules/retrieval/index/mistral_baseline_evaluation.py` - Baseline model evaluation
- `textwave/modules/retrieval/index/index_strategy_comparison.py` - Index strategy analysis

### Testing Tools
- `quality_testing.py` - Quality assurance testing
- `textwave/test_qa.py` - Question answering testing
- `debug_tfidf.py` - TF-IDF debugging utilities

## Technologies Used

- **Backend**: Flask
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Search**: FAISS (Brute-Force, HNSW, LSH)
- **LLM**: Mistral AI API
- **Reranking**: Cross-encoders, TF-IDF, BoW
- **Data Processing**: NumPy, scikit-learn

## Performance Highlights

Key findings from benchmarking:
- **HNSW indexing** provides 10-100x speedup over brute-force with minimal accuracy loss
- **Cross-encoder reranking** improves answer relevance by 15-25%
- **Optimal chunk size** varies by document type (400-600 tokens for most documents)
- **Sequential reranking** (BoW → TF-IDF → Cross-encoder) provides best results

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
- Python 3.8+
- Flask
- sentence-transformers
- faiss-cpu
- mistralai
- numpy
- scikit-learn
