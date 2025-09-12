import math
import numpy as np
import re
from collections import Counter, defaultdict
from typing import List

class TF_IDF:
    """
    A TF-IDF transformer that learns a vocabulary from a corpus and transforms
    documents into their TF-IDF representation.
    
    The TF-IDF (Term Frequency - Inverse Document Frequency) model is a numerical
    statistic intended to reflect how important a word is to a document in a corpus.
    This transformer builds a vocabulary of unique tokens from the provided corpus and
    computes an IDF score for each token. New documents can then be transformed into
    a dictionary where each token is associated with its TF-IDF score.
    
    Attributes:
        vocabulary_ (dict): A mapping of tokens (str) to unique indices (int), created
                            during the fitting process.
        idf_ (numpy.ndarray): An array of IDF values indexed by vocabulary indices.
    """

    def __init__(self):
        """
        Initializes the TF_IDF transformer with an empty vocabulary and IDF mapping.
        
        This constructor sets up the transformer without any pre-loaded vocabulary.
        The vocabulary and IDF values will be computed when the 'fit' method is called.
        """
        self.vocabulary_ = {}
        self.idf_ = None  # Will be numpy array after fit

    def _tokenize(self, text: str):
        """
        Tokenizes the input text by converting it to lowercase and extracting words.
        
        The function uses a regular expression to match word boundaries and extract
        alphanumeric sequences as tokens. This is a basic tokenization approach that
        may be extended for more complex use cases.

        NOTE: We should exclude stop words!
        
        Parameters:
            text (str): The text to tokenize.
            
        Returns:
            list: A list of word tokens (str) extracted from the input text.
            
        Example:
            >>> tokens = TF_IDF()._tokenize("Hello World!")
            >>> print(tokens)
            ['hello', 'world']
        """
        # Convert to lowercase
        text = text.lower()
        
        # Use regex to extract alphanumeric words
        tokens = re.findall(r'\b\w+\b', text)
        
        return tokens

    def fit(self, documents: List[str]):
        """
        Learns the vocabulary and computes the inverse document frequency (IDF) from the corpus.
        
        The 'fit' method processes each document in the provided corpus, tokenizes them,
        and constructs a set of unique tokens. It then calculates the document frequency
        for each token (i.e., the number of documents that contain the token). The IDF for
        each token is computed using the formula:
        
            IDF(token) = log(total_documents / (document_frequency + 1)) + 1
        
        The vocabulary is stored as a mapping from token to index, and the IDF values
        are stored as a numpy array.
        
        Parameters:
            documents (list of str): A list of documents (each document is a string)
                                     that forms the training corpus.
                                     
        Returns:
            TF_IDF (object): The instance of the TF_IDF transformer with the learned vocabulary and IDF values.
            In other words, your function should end in: `return self`.
            
        Example:
            >>> corpus = ["The quick brown fox.", "Lazy dog."]
            >>> transformer = TF_IDF().fit(corpus)
        """
        # Collect all unique words and count document frequencies
        all_words = set()
        document_freq = defaultdict(int)
        
        # First pass: collect unique words and count document frequencies
        for document in documents:
            tokens = self._tokenize(document)
            unique_tokens_in_doc = set(tokens)
            
            # Add to overall vocabulary
            all_words.update(unique_tokens_in_doc)
            
            # Count document frequency for each unique token in this document
            for token in unique_tokens_in_doc:
                document_freq[token] += 1
        
        # Create sorted vocabulary mapping
        sorted_words = sorted(all_words)
        self.vocabulary_ = {word: index for index, word in enumerate(sorted_words)}
        
        # Calculate IDF for each token and store as numpy array
        total_documents = len(documents)
        idf_values = np.zeros(len(sorted_words))
        
        for i, token in enumerate(sorted_words):
            # IDF formula: log(total_documents / (document_frequency + 1)) + 1
            idf_value = math.log(total_documents / (document_freq[token] + 1)) + 1
            idf_values[i] = idf_value
        
        # Store IDF as numpy array (this is crucial for tests)
        self.idf_ = idf_values
        
        return self

    def transform(self, document: str):
        """
        Transforms a document into its TF-IDF representation.
        
        This method tokenizes the input document and computes the term frequency (TF) for each token.
        The TF is normalized by dividing the token count by the total number of tokens in the document.
        Each token's TF value is then multiplied by its corresponding IDF value (learned during 'fit')
        to obtain the TF-IDF score. Only tokens present in the learned vocabulary are included.
        
        Parameters:
            document (str): A single document (string) to be transformed.
            
        Returns:
            numpy.array: A numpy array indexing each term (from the learned vocabulary) with its TF-IDF scores in the document.
                  Only tokens present in the vocabulary are included.
            
        Example:
            >>> transformer = TF_IDF().fit(["The quick brown fox.", "Lazy dog."])
            >>> tfidf_vector = transformer.transform("The quick fox.")
        """
        # Check if fit has been called
        if not hasattr(self, 'vocabulary_') or not self.vocabulary_ or self.idf_ is None:
            raise AttributeError("This TF_IDF instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        # Initialize TF-IDF vector
        tfidf_vector = np.zeros(len(self.vocabulary_))
        
        # Tokenize the document
        tokens = self._tokenize(document)
        
        if not tokens:  # Handle empty document
            return tfidf_vector
        
        # Count token frequencies
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        # Calculate TF-IDF for each token
        for token, count in token_counts.items():
            if token in self.vocabulary_:
                # Term Frequency (normalized)
                tf = count
                
                # Get vocabulary index
                index = self.vocabulary_[token]
                
                # Get IDF value from numpy array
                idf = self.idf_[index]
                
                # Calculate TF-IDF
                tfidf_score = tf * idf
                
                # Set in vector at appropriate index
                tfidf_vector[index] = tfidf_score
        
        return tfidf_vector


if __name__ == "__main__":
    # Example corpus of 9 documents to fit the TF-IDF transformer.
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Never jump over the lazy dog quickly.",
        "A quick movement of the enemy will jeopardize six gunboats.",
        "All that glitters is not gold.",
        "To be or not to be, that is the question.",
        "I think, therefore I am.",
        "The only thing we have to fear is fear itself.",
        "Ask not what your country can do for you; ask what you can do for your country.",
        "That's one small step for man, one giant leap for mankind.",
    ]

    # Fit the transformer on the corpus.
    transformer = TF_IDF()
    transformer.fit(corpus)
    
    # Test document to transform after fitting the corpus.
    test_document = "The quick dog jumps high over the lazy fox."
    tfidf_test = transformer.transform(test_document)
    
    print("Vocabulary size:", len(transformer.vocabulary_))
    print("TF-IDF vector shape:", tfidf_test.shape)
    print("Non-zero TF-IDF scores:", [(i, score) for i, score in enumerate(tfidf_test) if score > 0])
