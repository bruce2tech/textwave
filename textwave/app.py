from flask import Flask, request, jsonify
import os
import glob
import numpy as np
from sentence_transformers import SentenceTransformer

# Import the modules we need
from modules.extraction.preprocessing import DocumentProcessing
from modules.retrieval.index.bruteforce import FaissBruteForce
from modules.retrieval.index.hnsw import FaissHNSW
from modules.retrieval.index.lsh import FaissLSH
from modules.retrieval.search import FaissSearch
from modules.retrieval.reranker import Reranker
from modules.generator.question_answering import QAGeneratorMistral

app = Flask(__name__)

#######################################
# DEFAULT SYSTEM PARAMETERS 
#######################################
STORAGE_DIRECTORY = "storage/"
CORPUS_DIRECTORY = "storage/"
CHUNKING_STRATEGY = 'fixed_length'  # or 'sentence'
CHUNKING_PARAMETERS = {
    "chunk_size": 500,  # Reasonable chunk size for documents
    "overlap_size": 50   # Some overlap for context preservation
}
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
INDEX_STRATEGY = "bruteforce"  # or "hnsw", "lsh"
INDEX_PARAMETERS = {}
RERANKING_STRATEGY = "tfidf"  # or "cross_encoder", "bow", "sequential"
RERANKING_PARAMETERS = {}

# Global variables to store the initialized components
faiss_index = None
embedding_model = None
reranker = None
qa_generator = None

def initialize_embedding_model():
    """Initialize the sentence transformer model for embeddings."""
    global embedding_model
    if embedding_model is None:

        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return embedding_model

def initialize_qa_generator():
    """Initialize the QA generator with Mistral API."""
    global qa_generator
    if qa_generator is None:
        api_key = os.environ.get("MISTRAL_API_KEY")
        print(f"DEBUG: API key found: {'Yes' if api_key else 'No'}")
        print(f"DEBUG: API key length: {len(api_key) if api_key else 0}")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        qa_generator = QAGeneratorMistral(api_key=api_key)
    return qa_generator

def initialize_reranker():
    """Initialize the reranker."""
    global reranker
    if reranker is None:
        reranker = Reranker(type=RERANKING_STRATEGY, **RERANKING_PARAMETERS)
    return reranker

def initialize_index():
    """
    1. Parse through all the documents contained in storage/corpus directory
    2. Chunk the documents using either a'sentence' and 'fixed-length' chunking strategies (indicated by the CHUNKING_STRATEGY value):
        NOTE: The CHUNKING_STRATEGY will configure either fixed chunk or sentence chunking
    3. Embed each chunk using Embedding class, using 'all-MiniLM-L6-v2' text embedding model as default.
    4. Store vector embeddings of these chunks in a FAISS index, along with the chunks as metadata. 
        NOTE: You will decide the best strategy. Use `bruteforce` as default.
    5. This function should return the FAISS index
    """
    global faiss_index
    
    if faiss_index is not None:
        return faiss_index
    
    print("Initializing index...")

    # Initialize components FIRST
    doc_processor = DocumentProcessing()
    embedding_model = initialize_embedding_model()
    
    # Try multiple possible paths for corpus files
    possible_paths = [
        CORPUS_DIRECTORY,  # Default: "storage/"
        "textwave/storage/",  # For tests running from project root
        "../textwave/storage/",  # For tests running from test directory
        "storage/",
        os.path.join(os.getcwd(), "storage/"),
        os.path.join(os.path.dirname(__file__), "storage/")
    ]
    
    corpus_files = []
    corpus_directory = None
    
    for path in possible_paths:
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, "*.txt.clean"))
            if files:
                corpus_files = files
                corpus_directory = path
                break
    
    if not corpus_files:
        raise FileNotFoundError(f"No .txt.clean files found in any of these paths: {possible_paths}")
    
    print(f"Found {len(corpus_files)} documents in {corpus_directory}")
    
    # Step 2: Process and chunk documents
    all_chunks = []
    all_metadata = []
    
    for file_path in corpus_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
        if CHUNKING_STRATEGY == 'fixed_length':
            chunks = doc_processor.fixed_length_chunking(
                file_path,
                chunk_size=CHUNKING_PARAMETERS["chunk_size"],
                overlap_size=CHUNKING_PARAMETERS["overlap_size"]
            )
        elif CHUNKING_STRATEGY == 'sentence':
            chunks = doc_processor.sentence_chunking(
                file_path,
                num_sentences=CHUNKING_PARAMETERS.get("num_sentences", 5),
                overlap_size=CHUNKING_PARAMETERS.get("overlap_size", 1)
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {CHUNKING_STRATEGY}")
        
        # Add chunks and metadata
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only add non-empty chunks
                all_chunks.append(chunk)
                all_metadata.append({
                    "source_file": filename,
                    "chunk_index": i,
                    "text": chunk
                })
    
    print(f"Created {len(all_chunks)} chunks total")
    
    # Step 3: Generate embeddings for all chunks
    print("Generating embeddings...")
    embeddings = embedding_model.encode(all_chunks, convert_to_numpy=True)
    embedding_dim = embeddings.shape[1]
    
    print(f"Generated embeddings with dimension {embedding_dim}")
    
    # Step 4: Create and populate FAISS index
    print(f"Creating {INDEX_STRATEGY} index...")
    
    if INDEX_STRATEGY == "bruteforce":
        faiss_index = FaissBruteForce(dim=embedding_dim, metric="cosine")
    elif INDEX_STRATEGY == "hnsw":
        faiss_index = FaissHNSW(dim=embedding_dim, metric="cosine", **INDEX_PARAMETERS)
    elif INDEX_STRATEGY == "lsh":
        faiss_index = FaissLSH(dim=embedding_dim, **INDEX_PARAMETERS)
    else:
        raise ValueError(f"Unsupported index strategy: {INDEX_STRATEGY}")
    
    # Add embeddings and metadata to index
    faiss_index.add_embeddings(embeddings, all_metadata)
    
    print("Index initialization complete!")
    return faiss_index

@app.route("/generate", methods=["POST"])
def generate_answer():
    """
    Generate an answer to a given query by running the retrieval and reranking pipeline.

    This endpoint accepts a POST request with a JSON body containing the "query" field.
    It preprocesses and indexes the corpus if necessary, retrieves top-k relevant documents,
    and uses a language model to generate a final answer.

    Example curl command:
    curl -X POST http://localhost:5000/generate \
         -H "Content-Type: application/json" \
         -d '{"query": "What is the role of antioxidants in green tea?"}'

    :return: JSON response containing the generated answer.
    """
    try:
        # Check for JSON content type
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        # Get query from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must contain valid JSON"}), 400
            
        if "query" not in data:
            return jsonify({"error": "Query field is required"}), 400
        
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 422
        
        # Handle very long queries gracefully
        if len(query) > 10000:  # Reasonable limit
            query = query[:10000]  # Truncate instead of failing
            
        print(f"Processing query: {query[:100]}...")  # Truncate log output
        
        # Initialize components if needed
        try:
            index = initialize_index()
            embedding_model = initialize_embedding_model()
            reranker = initialize_reranker()
            qa_generator = initialize_qa_generator()
        except Exception as init_error:
            print(f"Initialization error: {init_error}")
            return jsonify({"error": "Service initialization failed. Please check corpus files and configuration."}), 500

        # Step 1: Embed the query
        query_embedding = embedding_model.encode(query)
        
        # Ensure proper shape for search
        if hasattr(query_embedding, 'cpu'):  # PyTorch tensor
            query_embedding = query_embedding.cpu().numpy()
        
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Step 2: Search the index for relevant documents
        search = FaissSearch(index, metric='cosine')
        distances, indices, metadata_results = search.search(query_embedding, k=20)
        
        # Extract document texts from metadata
        retrieved_docs = []
        for metadata_list in metadata_results:
            for metadata in metadata_list:
                if metadata and "text" in metadata:
                    retrieved_docs.append(metadata["text"])
        
        print(f"Retrieved {len(retrieved_docs)} candidate documents")
        
        # Step 3: Rerank the retrieved documents
        if retrieved_docs:
            reranked_docs, reranked_indices, scores = reranker.rerank(query, retrieved_docs)
            
            # Take top 5 documents for context
            top_context = reranked_docs[:5]
            print(f"Using top {len(top_context)} documents for answer generation")
        else:
            top_context = []
        
        # Step 4: Generate answer using the QA generator
        if top_context:
            try:
                # Calculate total context length
                total_context_length = sum(len(doc) for doc in top_context)
                print(f"DEBUG: Total context length: {total_context_length} characters")
                
                # Ensure context is reasonable length for API
                if total_context_length > 6000:
                    # Truncate context
                    truncated_context = []
                    current_length = 0
                    max_length = 5000
                    
                    for doc in top_context:
                        if current_length + len(doc) > max_length:
                            remaining_space = max_length - current_length
                            if remaining_space > 100:
                                truncated_context.append(doc[:remaining_space] + "...")
                            break
                        else:
                            truncated_context.append(doc)
                            current_length += len(doc)
                    
                    final_context = truncated_context
                    print(f"DEBUG: Using {len(final_context)} documents, total length: {sum(len(doc) for doc in final_context)}")
                else:
                    final_context = top_context
                
                answer = generate_answer_with_requests(query, final_context)
                print(f"DEBUG: Successfully generated answer")
                
            except Exception as e:
                print(f"DEBUG: Mistral API call failed: {e}")
                # Provide fallback with retrieved content
                answer = f"I found relevant information about your question: {top_context[0][:400]}..."
                
        else:
            answer = "I couldn't find relevant information to answer your question."

        return jsonify({
            "query": query, 
            "answer": answer,
            "num_documents_retrieved": len(retrieved_docs),
            "num_documents_used": len(top_context)
        })
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

def generate_answer_with_requests(query, context):
    """Alternative QA generation using requests instead of Mistral SDK"""
    import requests
    
    api_key = os.environ.get("MISTRAL_API_KEY")
    
    # Debug: Show what context is being sent
    print(f"DEBUG: Sending {len(context)} documents to Mistral:")
    for i, doc in enumerate(context):
        print(f"  Doc {i+1} ({len(doc)} chars): {doc[:100]}...")
    
    combined_input = (
        f"Question: {query}\n\n"
        f"Context: {', '.join(context)}\n\n"
    )
    
    # Debug: Show the combined input
    print(f"DEBUG: Combined input being sent:")
    print(f"  {combined_input[:500]}...")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "mistral-small-latest",
        "temperature": 0.3,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer the user's question using ONLY the provided context. "
                    "If the context contains relevant information, provide a helpful answer. "
                    "If the context does not contain relevant information, say 'No relevant information found in the provided context.'"
                )
            },
            {
                "role": "user",
                "content": combined_input,
            },
        ]
    }
    
    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            print(f"DEBUG: Mistral returned: {answer}")
            return answer
        else:
            return f"API Error: {response.status_code}"
            
    except Exception as e:
        return f"Request failed: {e}"

@app.route("/upload", methods=["POST"])
def upload_document():
    """
    Upload a new document to the corpus and rebuild the index.

    This endpoint accepts a POST request with a JSON body containing:
    - "text": The document text to add (required)
    - "filename": Optional filename (will be auto-generated if not provided)

    Example curl command:
    curl -X POST http://localhost:5000/upload \
         -H "Content-Type: application/json" \
         -d '{"text": "Green tea contains antioxidants called catechins..."}'

    :return: JSON response confirming the upload.
    """
    global faiss_index

    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Request body must contain 'text' field"}), 400

        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 422

        # Generate filename if not provided
        filename = data.get("filename", "").strip()
        if not filename:
            import time
            filename = f"upload_{int(time.time())}.txt.clean"
        elif not filename.endswith(".txt.clean"):
            filename = filename + ".txt.clean"

        # Find the storage directory
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "storage/"),
            "textwave/storage/",
            "storage/",
        ]

        storage_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                storage_dir = path
                break

        if not storage_dir:
            # Create storage directory if it doesn't exist
            storage_dir = os.path.join(os.path.dirname(__file__), "storage/")
            os.makedirs(storage_dir, exist_ok=True)

        # Save the document
        file_path = os.path.join(storage_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Saved document to {file_path}")

        # Reset the index so it will be rebuilt with the new document
        faiss_index = None

        # Reinitialize the index
        initialize_index()

        return jsonify({
            "status": "success",
            "message": f"Document uploaded and index rebuilt",
            "filename": filename,
            "text_length": len(text)
        })

    except Exception as e:
        print(f"Error uploading document: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    print("Starting TextWave Flask Application...")
    print(f"Corpus directory: {CORPUS_DIRECTORY}")
    print(f"Chunking strategy: {CHUNKING_STRATEGY}")
    print(f"Index strategy: {INDEX_STRATEGY}")
    print(f"Reranking strategy: {RERANKING_STRATEGY}")
    
    app.run(host="0.0.0.0", port=5000, debug=True)