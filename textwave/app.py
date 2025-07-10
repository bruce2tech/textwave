from flask import Flask, request, jsonify
# TODO: Add your import statements


# TODO: You will need to implement: 
# - initialize_index()
# - generate_answer()

app = Flask(__name__)


#######################################
# DEFAULT SYSTEM PARAMETERS 
# TODO: Define your default parameters 
# here. You will decide the optimal 
# strategies for each module.
# Format these parameters using the 
# following template...
#######################################
STORAGE_DIRECTORY = "storage/"
CHUNKING_STRATEGY = 'fixed-length' # or 'sentence'
CHUNKING_PARAMETERS = {
    "chunk_size": 10, 
    "overlap_size": 0
}
EMBEDDING_MODEL = 'all-MiniLM-L6-v2' # 
INDEX_STRATEGY = "bruteforce"
INDEX_PARAMETERS = {}
# add more as needed...
# RERANKING_STRATEGY = None 
# RERANKING_PARAMETERS = {}
# ...



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

    #######################################
    # TODO: Implement initialize()
    #######################################
    pass




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
    answer = None
    query = None
    #######################################
    # TODO: Implement generate_answer()
    #######################################

    return jsonify({"query": query, "answer": answer})

if __name__ == "__main__":
    app.run(debug=True)







