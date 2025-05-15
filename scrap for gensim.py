from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

sentences = [
    "This is the first sentence.",
    "This is the second sentence.",
    "And this is the third sentence."
]

tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4, sg=0)

vector = model.wv['sentence']
similar_words = model.wv.most_similar('sentence', topn=3)

print("Vector representation of 'sentence':", vector)
print("Words similar to 'sentence':", similar_words)
