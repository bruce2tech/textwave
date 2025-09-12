import re
from collections import Counter
from typing import List
import numpy as np

# from text_processing import process_text

# NOTE: Efficiency is not the primary goal here; nevertheless, 
#       using :class:`collections.Counter` is recommended. See 
#       the following resources for more information: 
#       - https://docs.python.org/3/library/collections.html
#       - https://www.geeksforgeeks.org/python/counters-in-python-set-1/

class BagOfWords:
    """
    A Bag-of-Words represnation transformer that learns a vocabulary from a corpus and transforms
    documents into their Bag-of-Words (BoW) representation.

    The BoW model represents text data as a collection of word counts, ignoring the
    order and structure of words. This transformer builds a vocabulary from the provided
    training corpus and then counts occurrences of these vocabulary words in new documents.
    """

    def __init__(self):
        """
        Initializes the Bag_of_Words transformer with an empty vocabulary.

        Attributes:
            vocabulary_ (dict): A dictionary mapping each unique word found in the corpus
                                to a unique index. This is constructed during the fit process.
        """
        self.vocabulary_ = {}

    def _tokenize(self, text: str):
        """
        Tokenizes the input text by converting it to lowercase and extracting words using a regular expression.

        This basic tokenization approach splits the text on word boundaries, capturing only alphanumeric
        sequences. Adjust the regular expression if you require a different tokenization strategy.

        Parameters:
            text (str): The input text to be tokenized.

        Returns:
            list: A list of word tokens extracted from the text.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Use regex to extract alphanumeric words
        tokens = re.findall(r'\b\w+\b', text)
        
        return tokens

    def fit(self, documents: List[str]):
        """
        Learns the vocabulary from the corpus by processing each document and extracting unique tokens.

        During this process, each document in the training corpus is tokenized, and the set of unique
        words is aggregated across all documents. The vocabulary is then created by sorting these unique words
        and assigning each a unique index.

        Parameters:
            documents (list of str): The training corpus where each document is a string.

        Returns:
            Bag_of_Words: The fitted transformer instance with an updated vocabulary_ attribute.
        """
        # Collect all unique words from all documents
        all_words = set()
        
        for document in documents:
            tokens = self._tokenize(document)
            all_words.update(tokens)
        
        # Sort the unique words and create vocabulary mapping
        sorted_words = sorted(all_words)
        self.vocabulary_ = {word: index for index, word in enumerate(sorted_words)}
        
        return self
    
    def transform(self, document: str):
        """
        Transforms a single document into its Bag-of-Words representation.

        This method tokenizes the input document and counts the occurrences of each token that exists
        in the learned vocabulary. The output is a numpy array indexed by ordered tokens (words) and values
        are their corresponding counts in the document.

        Parameters:
            document (str): A single document to be transformed into a BoW vector.

        Returns:
            numpy: A numpy array indexing each term (from the learned vocabulary) with its count in the document.
                  Only tokens present in the vocabulary are included.
        """
        # Check if fit has been called (vocabulary should exist and not be empty)
        if not hasattr(self, 'vocabulary_') or not self.vocabulary_:
            raise AttributeError("This BagOfWords instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        # Initialize a zero vector with length equal to vocabulary size
        bow_vector = np.zeros(len(self.vocabulary_))
        
        # Tokenize the document
        tokens = self._tokenize(document)
        
        # Count occurrences of each token
        token_counts = Counter(tokens)
        
        # Fill the bow vector with counts for tokens that exist in vocabulary
        for token, count in token_counts.items():
            if token in self.vocabulary_:
                index = self.vocabulary_[token]
                bow_vector[index] = count
        
        return bow_vector



if __name__ == "__main__":
    # Example corpus of 9 documents to train the Bag-of-Words representation.
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

    # Fit the transform on the corpus.
    transform = BagOfWords()
    transform.fit(corpus)
    
    # Test document to transform after fitting the corpus.
    test_document = "The quick dog jumps high over the lazy fox."
    bow_test = transform.transform(test_document)
    
    print("Vocabulary:", transform.vocabulary_)
    print("BoW vector:", bow_test)
    print("Non-zero elements:", [(i, count) for i, count in enumerate(bow_test) if count > 0])