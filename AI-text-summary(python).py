import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize #used for analyzing words and sentences.

nltk.download("punkt") #used to split the text into sentences
nltk.download("stopwords") # used to remove words like the, and, is

def understand_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words("english"))

    words = [word_tokenize(sentence.lower()) for sentence in sentences]
    words = [[word for word in sentence if word.isalnum() and word not in stop_words] for sentence in words]

    return sentences, words

text = "Hello, this is a example sentence"
sentences, words = understand_text(text)

print("sentences:", sentences)
print("changes words:", words)
