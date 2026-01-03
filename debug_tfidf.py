# Debug TF-IDF calculation
import sys
sys.path.append('./textwave')
from modules.utils.tfidf import TF_IDF

# Use the same corpus from the failing test
corpus = ["alpha beta gamma", "beta gamma delta", "gamma delta epsilon"]
tfidf = TF_IDF()
tfidf.fit(corpus)

doc = "alpha beta"
vec = tfidf.transform(doc)

print("Corpus:", corpus)
print("Document:", doc)
print("Vocabulary:", tfidf.vocabulary_)
print("IDF values:", tfidf.idf_)
print("TF-IDF vector:", vec)
print("Non-zero values:", {i: vec[i] for i in range(len(vec)) if vec[i] > 0})