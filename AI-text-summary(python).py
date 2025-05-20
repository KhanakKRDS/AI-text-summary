import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize #used for analyzing words and sentences.
from collections import Counter
from flask import Flask, request, render_template #helpful for website

##nltk.download("punkt") #used to split the text into sentences
##nltk.download("stopwords") # used to remove words like the, and, is

app = Flask(_name_)

def understand_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words("english"))

    words = [word_tokenize(sentence.lower()) for sentence in sentences]
    words = [[word for word in sentence if word.isalnum() and word not in stop_words] for sentence in words]

    return sentences, words

text =("Hello, this is a example sentence")
sentences, words = understand_text(text)

print("sentences:", sentences)
print("changes words:", words)

def important_words(words):
    word_counts = Counter([word for sentence in words for word in sentence])# count how many times the word occurred
    
    return word_counts

word_freq = important_words(words)
print("Word Frequency:", word_freq)

def get_sentence_scores(sentences, word_freq):
    sentence_scores = {}

    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        sentence_scores[sentence] = sum(word_freq[word] for word in words if word in word_freq)

    return sentence_scores

sentence_scores = get_sentence_scores(sentences, word_freq)
print("sentence scores:", sentence_scores)


def get_summary(sentence_scores, top_n=2):
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse = True)
    summary_sentences = [sentence for sentence, in sorted_sentences[:top_n]]
    return"". join(summary_sentences)

@app.route("/", methods=["GET", "Post"])
def index():
    summary, word_feq = "", {}

    if request.method == "Post":
        text = request.form["text"]
        sentences, words = understand_text(text)
        word_freq = important_words(words)
        sentence_scores = get_sentence_scores(sentences, word_freq)
        summary = get_summary(sentence_scores)

    return render_template("AI text summary.html", summary = summary, word_freq = word_freq)

if _name_ == "_main_":
    app.run(debug=True)

summary = get_summary(sentence_scores)
print("Summary:", summary)



